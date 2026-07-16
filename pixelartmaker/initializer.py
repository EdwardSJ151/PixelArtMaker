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


def _remove_white_background(
    image: Image.Image, white_threshold: int = 240
) -> tuple[Image.Image, np.ndarray]:
    """BFS flood-fill from all four corners to identify pixels that are white (or near-white)
    AND connected to the image border.  Those pixels are treated as background.

    The dark sprite border stops the flood-fill from leaking into the foreground.

    Returns ``(cleaned_image, is_bg_mask)`` where background pixels in *cleaned_image*
    have been replaced with black (consistent with how alpha removal works).
    """
    from collections import deque

    pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = pixels.shape[:2]

    is_bg = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])

    while queue:
        y, x = queue.popleft()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        visited[y, x] = True
        r, g, b = int(pixels[y, x, 0]), int(pixels[y, x, 1]), int(pixels[y, x, 2])
        if r >= white_threshold and g >= white_threshold and b >= white_threshold:
            is_bg[y, x] = True
            queue.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

    result = pixels.copy()
    result[is_bg] = np.array([0, 0, 0], dtype=np.uint8)

    print(
        f"[bg] White-bg removal (threshold={white_threshold}): "
        f"masked {is_bg.sum()} background pixels"
    )
    return Image.fromarray(result), is_bg


def _flatten_background(
    image: Image.Image, tolerance: int = 40, white_threshold: int = 240
) -> tuple[Image.Image, np.ndarray, bool]:
    """Remove background, replace with uniform color.

    Priority order:
    1. Alpha channel — used when the image has transparency.
    2. White-background removal — used when >30 % of border pixels are near-white
       (i.e. all R, G, B channels >= white_threshold).  Relies on the sprite having
       a darker border that stops the flood-fill from entering the foreground.
    3. Generic BFS flood-fill from corners — fallback for all other cases.

    Returns ``(cleaned_image, is_bg_mask, used_alpha)`` where *is_bg_mask* is a
    boolean array (True = background pixel) and *used_alpha* indicates whether the
    alpha channel was the source of truth.
    """
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        alpha = np.array(rgba)[:, :, 3]
        is_bg = alpha < 128

        rgb = np.array(rgba)[:, :, :3]
        fill_color = np.array([0, 0, 0], dtype=np.uint8)
        result = rgb.copy()
        result[is_bg] = fill_color

        print(f"[bg] Alpha channel found — masked {is_bg.sum()} transparent pixels")
        return Image.fromarray(result), is_bg, True

    # Check whether the image has a predominantly white border before falling back
    # to the generic flood-fill.  Sample all four edges and count white-enough pixels.
    rgb_img = image.convert("RGB")
    border_pixels = np.array(rgb_img, dtype=np.uint8)
    h_b, w_b = border_pixels.shape[:2]
    top    = border_pixels[0, :, :]
    bottom = border_pixels[h_b - 1, :, :]
    left   = border_pixels[:, 0, :]
    right  = border_pixels[:, w_b - 1, :]
    border = np.concatenate([top, bottom, left, right], axis=0)
    white_border = np.all(border >= white_threshold, axis=1)
    white_fraction = white_border.mean()

    if white_fraction > 0.30:
        print(
            f"[bg] White border detected ({white_fraction:.1%} of border pixels >= {white_threshold}) "
            "— using white-background removal"
        )
        cleaned, is_bg = _remove_white_background(rgb_img, white_threshold=white_threshold)
        if not is_bg.any():
            print("[WARN] White-bg removal found no background pixels — skipping")
        return cleaned, is_bg, False

    pixels = np.array(rgb_img, dtype=np.uint8)
    is_bg = _flood_fill_background(pixels, tolerance)

    if not is_bg.any():
        print("[WARN] Background removal found no background pixels — skipping")
        return image.convert("RGB"), is_bg, False

    corners = [pixels[0, 0], pixels[0, pixels.shape[1] - 1],
               pixels[pixels.shape[0] - 1, 0], pixels[pixels.shape[0] - 1, pixels.shape[1] - 1]]
    fill_color = np.round(np.mean(corners, axis=0)).astype(np.uint8)

    result = pixels.copy()
    result[is_bg] = fill_color

    print(f"[bg] Flood-fill: flattened {is_bg.sum()} background pixels to {tuple(fill_color)}")
    return Image.fromarray(result), is_bg, False


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
    white_threshold: int = 240,
    lock_background: bool = False,
) -> PixelGrid:
    """Downsample image to grid_size×grid_size and map each pixel to the nearest palette color."""
    img = image.convert("RGB")
    locked = None
    if remove_background:
        img, bg_mask, _ = _flatten_background(
            img, tolerance=bg_tolerance, white_threshold=white_threshold
        )
        fg = ~bg_mask
        rows = np.any(fg, axis=1)
        cols = np.any(fg, axis=0)
        if rows.any():
            y1 = int(np.argmax(rows))
            y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
            x1 = int(np.argmax(cols))
            x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))
            img = img.crop((x1, y1, x2 + 1, y2 + 1))
            bg_mask = bg_mask[y1:y2 + 1, x1:x2 + 1]
            print(f"[bg] Cropped to foreground bounding box: {img.size[0]}×{img.size[1]} px")
        if lock_background:
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
