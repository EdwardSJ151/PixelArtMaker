"""Shared utilities used across multiple modules."""

from __future__ import annotations

import base64
import io
import os
import re

import numpy as np
from PIL import Image


def make_client(provider: str, base_url: str | None):
    """Build the LLM client for the given provider."""
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    from openai import OpenAI
    if provider == "vllm":
        return OpenAI(api_key="EMPTY", base_url=base_url or "http://localhost:8000/v1")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_tags(raw: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return _THINK_RE.sub("", raw).strip()


def img_to_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def img_to_b64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    return base64.b64encode(img_to_bytes(image)).decode("utf-8")


def fg_bounding_box(fg_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) cell bounds of the foreground region, or None if empty."""
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    if not rows.any():
        return None
    y1 = int(np.argmax(rows))
    y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x1 = int(np.argmax(cols))
    x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return x1, y1, x2, y2
