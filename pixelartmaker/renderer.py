"""Render a PixelGrid to a PIL Image."""

from __future__ import annotations

import io
import numpy as np
from PIL import Image

from .grid import PixelGrid
from .palette import _hex_to_rgb


def render(grid: PixelGrid, cell_size: int = 16) -> Image.Image:
    """Render the grid as a PIL Image with each cell scaled to cell_size×cell_size pixels."""
    h, w = grid.height, grid.width
    img = Image.new("RGB", (w * cell_size, h * cell_size))
    pixels = img.load()

    for y in range(h):
        for x in range(w):
            color_hex = grid.palette.hex_of(grid.get(y, x))
            r, g, b = _hex_to_rgb(color_hex)
            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[x * cell_size + dx, y * cell_size + dy] = (r, g, b)

    return img


def render_to_bytes(grid: PixelGrid, cell_size: int = 16) -> bytes:
    """Render grid and return raw PNG bytes."""
    img = render(grid, cell_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_gif(frames: list[Image.Image], path: str, duration_ms: int = 200) -> None:
    """Save a list of PIL Images as an animated GIF."""
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
