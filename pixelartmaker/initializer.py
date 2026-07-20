"""Convert an image to a PixelGrid, with optional LLM-based sprite size selection."""

from __future__ import annotations

import json
import re
from collections import deque

import numpy as np
from PIL import Image

from .grid import PixelGrid
from .palette import Palette
from .utils import make_client, strip_think_tags, img_to_bytes, img_to_b64, fg_bounding_box

_SPRITE_SIZE_LABELS = {
    "8x8":   "icon-scale sprite (minimal, single symbol or tile)",
    "8x16":  "NES tall character sprite (slim figure, one tile wide)",
    "16x16": "small character sprite (full character, simple detail)",
    "16x32": "tall character sprite (SNES portrait figure)",
    "32x32": "standard character sprite (full detail, character or enemy)",
    "32x64": "large tall sprite (SNES boss or tall character)",
    "48x48": "detailed character (complex design, large enemy)",
    "64x64": "large scene or boss sprite",
}

_RESAMPLE_MODES = {
    "nearest":  Image.NEAREST,
    "box":      Image.BOX,
    "lanczos":  Image.LANCZOS,
    "bilinear": Image.BILINEAR,
}


def select_sprite_size(
    image: Image.Image,
    description: str,
    client,
    model: str,
    provider: str,
    size_presets: list[str] | None = None,
) -> tuple[int, int]:
    """Ask the LLM to pick the best sprite size (WxH). Returns (width, height).
    Falls back to 32x32 on any error.
    """
    presets = size_presets or ["8x8", "16x16", "32x32", "48x48", "64x64"]
    default = "32x32" if "32x32" in presets else presets[len(presets) // 2]
    preset_lines = "\n".join(
        f"- {p}: {_SPRITE_SIZE_LABELS.get(p, 'custom size')}" for p in presets
    )
    prompt = (
        f'You are choosing a pixel art sprite resolution.\n'
        f'Description: "{description}"\n\n'
        f'Available sizes (WxH):\n{preset_lines}\n\n'
        f'Reply with ONLY valid JSON: {{"sprite_size": "{default}", "reason": "one sentence"}}'
    )

    raw = ""
    try:
        image_bytes = img_to_bytes(image)
        if provider == "gemini":
            from google.genai import types
            response = client.models.generate_content(
                model=model,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
            )
            raw = response.text
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(image)}"}},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
            )
            raw = response.choices[0].message.content

        raw = strip_think_tags(raw.strip())
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        size_str = str(data["sprite_size"]).lower().strip()
        if size_str not in presets:
            size_str = default
        w, h = (int(x) for x in size_str.split("x"))
        print(f"Sprite size selected: {w}×{h} — {data.get('reason', '')}")
        return w, h
    except Exception as e:
        print(f"[ERROR] Sprite size selection failed: {e}")
        if raw:
            print(f"[ERROR] Raw response: {raw!r}")
        w, h = (int(x) for x in default.split("x"))
        print(f"[ERROR] Defaulting to {w}×{h}")
        return w, h


# Keep old name as alias for any callers that haven't updated yet
def select_grid_size(
    image: Image.Image,
    description: str,
    client,
    model: str,
    provider: str,
    grid_presets: list[int] | None = None,
) -> int:
    presets_str = [f"{p}x{p}" for p in (grid_presets or [8, 16, 32, 48, 64])]
    w, h = select_sprite_size(image, description, client, model, provider, presets_str)
    return w


def _flood_fill_background(pixels: np.ndarray, tolerance: int) -> np.ndarray:
    """BFS flood-fill from corners. Returns bool mask (True = background)."""
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


def _remove_white_background(pixels: np.ndarray, white_threshold: int) -> np.ndarray:
    """BFS flood-fill from corners marking near-white pixels as background.
    Returns bool mask (True = background). The sprite's dark border stops the fill.
    """
    h, w = pixels.shape[:2]
    is_bg = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])

    while queue:
        y, x = queue.popleft()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        visited[y, x] = True
        if np.all(pixels[y, x] >= white_threshold):
            is_bg[y, x] = True
            queue.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

    return is_bg


def flatten_background(
    image: Image.Image,
    tolerance: int = 40,
    white_threshold: int = 240,
    second_white_threshold: int | None = None,
) -> tuple[Image.Image, np.ndarray, bool]:
    """Remove image background. Returns (cleaned_image, bg_mask, used_alpha).

    Priority:
    1. Alpha channel — if the image has transparency, use alpha < 128 as the mask.
    2. White background — if >30% of border pixels are near-white, use white BFS.
    3. Generic flood-fill from corners (fallback).

    If second_white_threshold is set, a second BFS white-removal pass runs after
    the first using the lower threshold, expanding the masked region.

    Background pixels are replaced with black in the returned image.
    used_alpha=True only when an alpha channel was the source of truth.
    """
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
        is_bg = rgba[:, :, 3] < 128
        result = rgba[:, :, :3].copy()
        result[is_bg] = 0
        print(f"[bg] Alpha channel — masked {is_bg.sum()} transparent pixels")
        if second_white_threshold is not None:
            second_mask = _remove_white_background(result, second_white_threshold)
            is_bg = is_bg | second_mask
            result[second_mask] = 0
            print(f"[bg] Second pass — masked {second_mask.sum()} additional near-white pixels")
        return Image.fromarray(result), is_bg, True

    pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = pixels.shape[:2]

    border = np.concatenate([
        pixels[0, :], pixels[h - 1, :], pixels[:, 0], pixels[:, w - 1]
    ])
    white_fraction = np.mean(np.all(border >= white_threshold, axis=1))

    if white_fraction > 0.30:
        print(f"[bg] White border ({white_fraction:.1%}) — using white-bg removal")
        is_bg = _remove_white_background(pixels, white_threshold)
    else:
        is_bg = _flood_fill_background(pixels, tolerance)

    if not is_bg.any():
        print("[WARN] Background removal found no background pixels — skipping")
        return Image.fromarray(pixels), is_bg, False

    result = pixels.copy()
    result[is_bg] = 0
    print(f"[bg] Masked {is_bg.sum()} background pixels")

    if second_white_threshold is not None:
        second_mask = _remove_white_background(result, second_white_threshold)
        second_mask = second_mask & ~is_bg
        if second_mask.any():
            is_bg = is_bg | second_mask
            result[second_mask] = 0
            print(f"[bg] Second pass — masked {second_mask.sum()} additional near-white pixels")

    return Image.fromarray(result), is_bg, False


def _downsample_mask(mask: np.ndarray, grid_width: int, grid_height: int) -> np.ndarray:
    """Downsample bool mask to grid_width×grid_height (majority vote via BOX)."""
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    small = mask_img.resize((grid_width, grid_height), Image.BOX)
    return np.array(small) > 127


def pixelate(
    image: Image.Image,
    grid_width: int,
    grid_height: int,
    palette: Palette,
    resample: str = "nearest",
    preflattened: tuple[Image.Image, np.ndarray] | None = None,
    lock_background: bool = False,
) -> PixelGrid:
    """Downsample image to grid_width×grid_height and snap each pixel to palette.

    If preflattened=(flat_image, bg_mask) is provided, uses those directly
    instead of re-running background removal.
    """
    locked = None

    if preflattened is not None:
        flat_img, bg_mask = preflattened
        img = flat_img.convert("RGB")
        bbox = fg_bounding_box(~bg_mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            img = img.crop((x1, y1, x2 + 1, y2 + 1))
            bg_mask = bg_mask[y1:y2 + 1, x1:x2 + 1]
            print(f"[bg] Cropped to foreground bounding box: {img.size[0]}×{img.size[1]} px")
        if lock_background:
            locked = _downsample_mask(bg_mask, grid_width, grid_height)
    else:
        img = image.convert("RGB")

    mode = _RESAMPLE_MODES.get(resample, Image.NEAREST)
    small = np.array(img.resize((grid_width, grid_height), mode), dtype=np.float32)

    pixels_flat = small.reshape(-1, 3)
    dists = np.sum((palette._rgb[None, :, :] - pixels_flat[:, None, :]) ** 2, axis=2)
    data = np.argmin(dists, axis=1).reshape(grid_height, grid_width).astype(np.int32)

    return PixelGrid(data, palette, locked=locked)
