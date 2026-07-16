"""Render a PixelGrid to a PIL Image."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .grid import PixelGrid
from .utils import img_to_bytes


def render(grid: PixelGrid, cell_size: int = 16) -> Image.Image:
    """Render the grid as a PIL Image. Each cell becomes a cell_size×cell_size block."""
    palette_rgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
         for h in grid.palette._hex_list],
        dtype=np.uint8,
    )
    rgb = palette_rgb[grid.data]  # (H, W, 3)
    tiled = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
    return Image.fromarray(tiled)


def render_with_ruler(grid: PixelGrid, cell_size: int = 16, tick_every: int = 0) -> Image.Image:
    """Render with coordinate ruler margins (col numbers top, row numbers left).

    tick_every=0 → auto: every 4 cells for grids >=16wide, every 2 otherwise.
    Numbers label the grid cell index (0-indexed), matching x/y in tool calls.
    """
    base = render(grid, cell_size)
    margin = 24
    w, h = base.size
    canvas = Image.new("RGB", (w + margin, h + margin), (240, 240, 240))
    canvas.paste(base, (margin, margin))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    step = tick_every if tick_every > 0 else (4 if grid.width >= 16 else 2)

    # column numbers across the top
    for col in range(0, grid.width, step):
        x = margin + col * cell_size + cell_size // 2
        label = str(col)
        bbox = font.getbbox(label)
        lw = bbox[2] - bbox[0]
        draw.text((x - lw // 2, 4), label, fill=(80, 80, 80), font=font)

    # row numbers down the left
    for row in range(0, grid.height, step):
        y = margin + row * cell_size + cell_size // 2
        label = str(row)
        bbox = font.getbbox(label)
        lh = bbox[3] - bbox[1]
        draw.text((2, y - lh // 2), label, fill=(80, 80, 80), font=font)

    return canvas


def render_ascii(grid: PixelGrid) -> tuple[str, dict[str, str]]:
    """Return (ascii_grid_str, legend) where legend maps color_name → 2-char code.

    Each color gets a unique 2-char code derived from its name.
    Grid is formatted as a text matrix with row/column indices.
    """
    names = grid.palette.names

    # assign unique 2-char codes
    codes: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        base = name[:2].lower()
        if base not in used:
            codes[name] = base
        else:
            # find a free code: first char + digit
            for i in range(10):
                candidate = name[0].lower() + str(i)
                if candidate not in used:
                    codes[name] = candidate
                    break
            else:
                # fallback: palette index
                idx = grid.palette.index_of(name)
                codes[name] = f"{idx:02d}"
        used.add(codes[name])

    # legend line
    legend_parts = [f"{codes[n]}={n}" for n in names]
    legend_line = "  ".join(legend_parts)

    # header row
    col_width = 4  # each cell takes 4 chars: "xy  "
    header = "     " + "".join(f"x{c:<3}" for c in range(grid.width))

    # data rows
    rows = [header]
    for y in range(grid.height):
        row_cells = "  ".join(codes[names[grid.data[y, x]]] for x in range(grid.width))
        rows.append(f"y{y:<3}:  {row_cells}")

    grid_str = (
        f"ASCII GRID (y=row, x=col, 0-indexed):\n"
        f"Legend: {legend_line}\n\n"
        + "\n".join(rows)
    )
    return grid_str, codes


def render_preview(
    grid: PixelGrid,
    affected_cells: list[tuple[int, int]],
    cell_size: int = 16,
    highlight_color: str = "#FF4444",
) -> Image.Image:
    """Render grid with a bright border drawn around each affected cell."""
    base = render(grid, cell_size).convert("RGBA")
    draw = ImageDraw.Draw(base)
    r, g, b = int(highlight_color[1:3], 16), int(highlight_color[3:5], 16), int(highlight_color[5:7], 16)
    border_color = (r, g, b, 255)
    for cx, cy in affected_cells:
        x0 = cx * cell_size
        y0 = cy * cell_size
        x1 = x0 + cell_size - 1
        y1 = y0 + cell_size - 1
        for offset in range(2):
            draw.rectangle(
                [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
                outline=border_color,
            )
    return base.convert("RGB")


def render_to_bytes(grid: PixelGrid, cell_size: int = 16) -> bytes:
    return img_to_bytes(render(grid, cell_size))


def save_gif(frames: list[Image.Image], path: str, duration_ms: int = 200) -> None:
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
