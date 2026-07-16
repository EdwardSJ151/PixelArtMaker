"""CLIP/SigLIP-based scoring for pixel art vs. text description."""

from __future__ import annotations

import math
from PIL import Image


class CLIPEvaluator:
    """Cosine similarity between a rendered image and a text description.

    Supports any open_clip model including SigLIP via hf-hub: prefix.
    For SigLIP models, pass model_name='hf-hub:timm/ViT-B-16-SigLIP' and
    leave pretrained empty — the hub spec is self-contained.
    """

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
            scale = math.ceil(224 / longest)  # smallest integer that gets us to ≥224px
            image = image.resize((w * scale, h * scale), Image.NEAREST)
        tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            return self._model.encode_image(tensor)

    def score(self, image: Image.Image, description: str) -> float:
        """Return cosine similarity in [-1, 1] between image and description."""
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
