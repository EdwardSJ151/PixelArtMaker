"""Scoring evaluators for pixel art vs. text description."""

from __future__ import annotations

import base64
import io
import math
import os
import re
from PIL import Image


class CLIPEvaluator:
    """Cosine similarity via open_clip (CLIP or SigLIP)."""

    def __init__(self, model_name: str = "hf-hub:timm/ViT-B-16-SigLIP", pretrained: str = ""):
        import open_clip
        is_hf_hub = model_name.startswith("hf-hub:")
        if is_hf_hub:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_name)
            self._tokenizer = open_clip.get_tokenizer(model_name)
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

    def score(self, image: Image.Image, description: str) -> float:
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
    """Uses the same vision LLM as the optimizer to score each step.

    Asks the model to rate how well the current pixel art matches the description
    on a 0-100 scale. Returns a 0.0-1.0 float.
    """

    def __init__(self, provider: str, model: str, base_url: str | None):
        self.provider = provider
        self.model = model
        self._client = self._init_client(provider, base_url)

    def _init_client(self, provider: str, base_url: str | None):
        if provider == "gemini":
            from google import genai
            return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            from openai import OpenAI
            if provider == "vllm":
                return OpenAI(api_key="EMPTY", base_url=base_url or "http://localhost:8000/v1")
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def score(self, image: Image.Image, description: str) -> float:
        """Return a 0.0-1.0 score from the VLM."""
        prompt = (
            f'Rate how well this pixel art matches the description: "{description}"\n\n'
            f"Score from 0 to 100 where:\n"
            f"0 = completely wrong subject/colors\n"
            f"50 = recognizable but missing key features\n"
            f"100 = matches the description perfectly given pixel art constraints\n\n"
            f"Reply with ONLY a single integer. No explanation."
        )

        try:
            image_bytes = self._image_to_bytes(image)

            if self.provider == "gemini":
                from google.genai import types
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
                )
                raw = response.text
            else:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]}],
                    max_tokens=256,
                )
                raw = response.choices[0].message.content

            # Strip <think> blocks
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            # Extract first number
            match = re.search(r"\d+", raw)
            if match:
                value = max(0, min(100, int(match.group())))
                return value / 100.0
            print(f"[WARN] VLM scorer returned no number: {raw[:100]!r}")
            return 0.0
        except Exception as e:
            print(f"[ERROR] VLM scorer failed: {e}")
            return 0.0
