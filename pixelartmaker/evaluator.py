"""CLIP-based scoring for pixel art vs. text description."""

from __future__ import annotations

import io
import numpy as np
from PIL import Image


class CLIPEvaluator:
    """Computes cosine similarity between a rendered image and a text description."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()
        self._text_features: "torch.Tensor | None" = None
        self._cached_description: str = ""

    def _encode_text(self, description: str):
        import torch
        tokens = self._tokenizer([description])
        with torch.no_grad():
            return self._model.encode_text(tokens)

    def _encode_image(self, image: Image.Image):
        import torch
        tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            return self._model.encode_image(tensor)

    def score(self, image: Image.Image, description: str) -> float:
        """Return cosine similarity in [0, 1] between image and description."""
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
