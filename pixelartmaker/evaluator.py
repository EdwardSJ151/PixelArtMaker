"""Scoring evaluators for pixel art vs. target image."""

from __future__ import annotations

import math
from PIL import Image

from .utils import make_client, strip_think_tags, img_to_bytes, img_to_b64


class CLIPEvaluator:
    """Cosine similarity via open_clip (CLIP or SigLIP)."""

    def __init__(self, model_name: str = "hf-hub:timm/ViT-B-16-SigLIP", pretrained: str = ""):
        import open_clip
        is_hf_hub = model_name.startswith("hf-hub:")
        if is_hf_hub:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_name)
        else:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()
        self._text_features = None
        self._cached_description: str = ""

    def _encode_text(self, description: str):
        import torch
        tokens = self._tokenizer([description])
        with torch.no_grad():
            return self._model.encode_text(tokens)

    def _encode_image(self, image: Image.Image):
        import torch
        w, h = image.size
        longest = max(w, h)
        if longest < 224:
            scale = math.ceil(224 / longest)
            image = image.resize((w * scale, h * scale), Image.NEAREST)
        tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            return self._model.encode_image(tensor)

    def score(self, image: Image.Image, description: str, **kwargs) -> float:
        import torch
        import torch.nn.functional as F
        if description != self._cached_description:
            self._text_features = self._encode_text(description)
            self._cached_description = description
        img_features = self._encode_image(image)
        sim = F.cosine_similarity(
            F.normalize(img_features, dim=-1),
            F.normalize(self._text_features, dim=-1),
        )
        return float(sim.item())


class VLMEvaluator:
    """Uses the vision LLM to score each step by comparing three images:
    original, current best, and candidate. Returns >0.5 if candidate wins.
    Returns 0.5 as baseline when no comparison images are available yet.
    """

    def __init__(self, provider: str, model: str, base_url: str | None):
        self.provider = provider
        self.model = model
        self._client = make_client(provider, base_url)

    def score(
        self,
        image: Image.Image,
        description: str,
        original: Image.Image | None = None,
        current_best: Image.Image | None = None,
    ) -> float:
        if original is None or current_best is None:
            return 0.5

        prompt = (
            f'You are evaluating pixel art edits. The goal is to match: "{description}"\n\n'
            f"You are shown three images in order:\n"
            f"1. ORIGINAL — the target pixel art to reproduce\n"
            f"2. CURRENT BEST — the best version so far\n"
            f"3. CANDIDATE — a new proposed edit\n\n"
            f"Compare CANDIDATE vs CURRENT BEST. Which is a closer match to ORIGINAL?\n\n"
            f"Reply with ONLY a number from 0 to 100 where:\n"
            f"0   = CURRENT BEST is clearly better\n"
            f"50  = they are equal\n"
            f"100 = CANDIDATE is clearly better\n\n"
            f"Single integer only. No explanation."
        )

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_to_bytes(original), mime_type="image/png"),
                        types.Part.from_bytes(data=img_to_bytes(current_best), mime_type="image/png"),
                        types.Part.from_bytes(data=img_to_bytes(image), mime_type="image/png"),
                    ],
                )
                raw = response.text
            else:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Image 1 — ORIGINAL:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(original)}"}},
                        {"type": "text", "text": "Image 2 — CURRENT BEST:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(current_best)}"}},
                        {"type": "text", "text": "Image 3 — CANDIDATE:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(image)}"}},
                    ]}],
                    max_tokens=256,
                )
                raw = response.choices[0].message.content

            raw = strip_think_tags(raw)
            import re
            match = re.search(r"\d+", raw)
            if match:
                return max(0, min(100, int(match.group()))) / 100.0
            print(f"[WARN] VLM scorer returned no number: {raw[:100]!r}")
            return 0.5
        except Exception as e:
            print(f"[ERROR] VLM scorer failed: {e}")
            return 0.5
