"""PixelGrid — the core data structure for a pixel art canvas."""

from __future__ import annotations

import numpy as np

from .palette import Palette


class PixelGrid:
    """A fixed-size 2D grid where each cell holds a palette index."""

    def __init__(self, data: np.ndarray, palette: Palette, locked: np.ndarray | None = None):
        """
        Args:
            data: H×W int32 array of palette indices.
            palette: Active palette for this grid.
            locked: H×W bool array — True = cell cannot be edited (background).
        """
        self.data = data.astype(np.int32)
        self.palette = palette
        self.locked = locked if locked is not None else np.zeros(data.shape, dtype=bool)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    def copy(self) -> "PixelGrid":
        return PixelGrid(self.data.copy(), self.palette, locked=self.locked.copy())

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

