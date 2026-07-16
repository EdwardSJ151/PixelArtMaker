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

    raw = ""
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
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=32768,
            )
            raw = response.choices[0].message.content

        raw = raw.strip()
        # Strip <think>...</think> blocks (Qwen reasoning models)
        import re as _re
        raw = _re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=_re.IGNORECASE).strip()
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
        print(f"[ERROR] Grid size selection failed: {e}")
        if raw:
            print(f"[ERROR] Raw LLM response was: {raw!r}")
        print(f"[ERROR] Defaulting to {default_size}×{default_size}")
        return default_size


_RESAMPLE_MODES = {
    "nearest":  Image.NEAREST,
    "box":      Image.BOX,
    "lanczos":  Image.LANCZOS,
    "bilinear": Image.BILINEAR,
}


def _flood_fill_background(pixels: np.ndarray, tolerance: int) -> np.ndarray:
    """BFS flood-fill from all four corners. Returns bool mask where True = background."""
    from collections import deque
    h, w = pixels.shape[:2]
    corners = [pixels[0, 0], pixels[0, w - 1], pixels[h - 1, 0], pixels[h - 1, w - 1]]
    bg_color = np.mean(corners, axis=0)

    is_bg = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])

    while queue:
        y, x = queue.popleft()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        visited[y, x] = True
        if np.sqrt(np.sum((pixels[y, x].astype(float) - bg_color) ** 2)) <= tolerance:
            is_bg[y, x] = True
            queue.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

    return is_bg


def _flatten_background(image: Image.Image, tolerance: int = 40) -> tuple[Image.Image, np.ndarray]:
    """Flood-fill background from corners, replace with uniform color.

    Returns the cleaned image AND the boolean background mask (True = background pixel).
    """
    pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    is_bg = _flood_fill_background(pixels, tolerance)

    if not is_bg.any():
        print("[WARN] Background removal found no background pixels — skipping")
        return image, is_bg

    corners = [pixels[0, 0], pixels[0, pixels.shape[1] - 1],
               pixels[pixels.shape[0] - 1, 0], pixels[pixels.shape[0] - 1, pixels.shape[1] - 1]]
    fill_color = np.round(np.mean(corners, axis=0)).astype(np.uint8)

    result = pixels.copy()
    result[is_bg] = fill_color

    print(f"[bg] Flattened {is_bg.sum()} background pixels to {tuple(fill_color)}")
    return Image.fromarray(result), is_bg


def _downsample_mask(mask: np.ndarray, grid_size: int) -> np.ndarray:
    """Downsample a full-res boolean mask to grid_size×grid_size.
    A cell is background if majority of its source pixels are background.
    """
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    small = mask_img.resize((grid_size, grid_size), Image.BOX)
    return np.array(small) > 127


def pixelate(
    image: Image.Image,
    grid_size: int,
    palette: Palette,
    resample: str = "nearest",
    remove_background: bool = False,
    bg_tolerance: int = 40,
) -> PixelGrid:
    """Downsample image to grid_size×grid_size and map each pixel to the nearest palette color."""
    img = image.convert("RGB")
    locked = None
    if remove_background:
        img, bg_mask = _flatten_background(img, tolerance=bg_tolerance)
        locked = _downsample_mask(bg_mask, grid_size)

    mode = _RESAMPLE_MODES.get(resample, Image.NEAREST)
    small = img.resize((grid_size, grid_size), mode)
    pixels = np.array(small)

    data = np.zeros((grid_size, grid_size), dtype=np.int32)
    for y in range(grid_size):
        for x in range(grid_size):
            r, g, b = int(pixels[y, x, 0]), int(pixels[y, x, 1]), int(pixels[y, x, 2])
            data[y, x] = palette.nearest_index(r, g, b)

    return PixelGrid(data, palette, locked=locked)
