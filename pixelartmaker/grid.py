"""PixelGrid — the core data structure for a pixel art canvas."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .palette import Palette


class PixelGrid:
    """A fixed-size 2D grid where each cell holds a palette index."""

    def __init__(self, data: np.ndarray, palette: Palette):
        """
        Args:
            data: H×W int32 array of palette indices.
            palette: Active palette for this grid.
        """
        self.data = data.astype(np.int32)
        self.palette = palette

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    def copy(self) -> "PixelGrid":
        return PixelGrid(self.data.copy(), self.palette)

    def get(self, y: int, x: int) -> int:
        return int(self.data[y, x])

    def set(self, y: int, x: int, index: int) -> None:
        self.data[y, x] = index

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def diff(self, other: "PixelGrid") -> int:
        """Number of cells that differ from another grid."""
        return int(np.sum(self.data != other.data))

    def change_ratio(self, other: "PixelGrid") -> float:
        return self.diff(other) / self.data.size

    def to_color_string(self) -> str:
        """Text representation for debugging (color names)."""
        rows = []
        for y in range(self.height):
            row = [self.palette.name_of(int(self.data[y, x]))[:3] for x in range(self.width)]
            rows.append(" ".join(row))
        return "\n".join(rows)

    @classmethod
    def blank(cls, height: int, width: int, palette: Palette, fill_index: int = 0) -> "PixelGrid":
        data = np.full((height, width), fill_index, dtype=np.int32)
        return cls(data, palette)
