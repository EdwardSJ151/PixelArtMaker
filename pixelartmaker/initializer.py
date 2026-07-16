"""Convert an image to a PixelGrid, with optional LLM-based grid size selection."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

from .grid import PixelGrid
from .palette import Palette

GRID_PRESETS = [8, 16, 32, 48, 64]

_PRESET_LABELS = {
    8:  "ultra-minimal icon (simple symbol, single emoji-like shape)",
    16: "small sprite (character, item, simple object)",
    32: "standard sprite (character with detail, simple scene)",
    48: "detailed sprite (complex character, small scene)",
    64: "large sprite (scene with background, high detail)",
}


def select_grid_size(
    image: Image.Image,
    description: str,
    client,
    model: str,
    provider: str,
    grid_presets: list[int] | None = None,
) -> int:
    """Ask the LLM to pick the best grid size for the image and description.

    Returns one of grid_presets. Falls back to 32 (or the nearest preset) on any error.
    """
    presets = grid_presets if grid_presets else GRID_PRESETS
    default_size = min(presets, key=lambda p: abs(p - 32))

    preset_lines = "\n".join(
        f"- {p}: {_PRESET_LABELS.get(p, 'custom size')}" for p in sorted(presets)
    )
    prompt = (
        f'You are choosing a pixel art grid resolution.\n'
        f'Description: "{description}"\n\n'
        f'Available presets:\n{preset_lines}\n\n'
        f'Reply with ONLY valid JSON: {{"grid_size": {default_size}, "reason": "one sentence"}}'
    )

    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        if provider == "gemini":
            from google.genai import types
            response = client.models.generate_content(
                model=model,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
            )
            raw = response.text
        else:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=64,
            )
            raw = response.choices[0].message.content

        # Parse JSON from response
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        size = int(data["grid_size"])
        size = min(presets, key=lambda p: abs(p - size))
        print(f"Grid size selected: {size}×{size} — {data.get('reason', '')}")
        return size
    except Exception as e:
        print(f"Grid size selection failed ({e}), defaulting to {default_size}×{default_size}")
        return default_size


def pixelate(image: Image.Image, grid_size: int, palette: Palette) -> PixelGrid:
    """Downsample image to grid_size×grid_size and map each pixel to the nearest palette color."""
    small = image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)
    pixels = np.array(small)

    data = np.zeros((grid_size, grid_size), dtype=np.int32)
    for y in range(grid_size):
        for x in range(grid_size):
            r, g, b = int(pixels[y, x, 0]), int(pixels[y, x, 1]), int(pixels[y, x, 2])
            data[y, x] = palette.nearest_index(r, g, b)

    return PixelGrid(data, palette)
