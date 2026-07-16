"""EditManager — central controller for all pixel edits."""

from __future__ import annotations

from collections import deque

from .grid import PixelGrid


class EditManager:
    """Validates and applies all edits to the PixelGrid."""

    def __init__(self, grid: PixelGrid):
        self._grid = grid.copy()

    @property
    def grid(self) -> PixelGrid:
        return self._grid.copy()

    def checkpoint(self) -> PixelGrid:
        return self._grid.copy()

    def rollback(self, checkpoint: PixelGrid) -> None:
        self._grid = checkpoint.copy()

    def apply(self, grid: PixelGrid) -> None:
        self._grid = grid.copy()

    def _resolve_color(self, color: str) -> int | None:
        """Return palette index for a color name, or None if invalid."""
        names = self._grid.palette.names
        if color in names:
            return names.index(color)
        return None

    def _is_locked(self, y: int, x: int) -> bool:
        return bool(self._grid.locked[y, x])

    def set_pixel(self, x: int, y: int, color: str) -> tuple[bool, str]:
        idx = self._resolve_color(color)
        if idx is None:
            return False, f"Unknown color '{color}'. Valid: {self._grid.palette.names}"
        if not self._grid.in_bounds(y, x):
            return False, f"Position ({x},{y}) out of bounds ({self._grid.width}×{self._grid.height})"
        if self._is_locked(y, x):
            return False, f"Cell ({x},{y}) is a locked background cell"
        self._grid.set(y, x, idx)
        return True, ""

    def set_rect(self, x1: int, y1: int, x2: int, y2: int, color: str, filled: bool = True) -> tuple[bool, str]:
        idx = self._resolve_color(color)
        if idx is None:
            return False, f"Unknown color '{color}'"
        y_min, y_max = min(y1, y2), max(y1, y2)
        x_min, x_max = min(x1, x2), max(x1, x2)
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if filled or y in (y_min, y_max) or x in (x_min, x_max):
                    if self._grid.in_bounds(y, x) and not self._is_locked(y, x):
                        self._grid.set(y, x, idx)
        return True, ""

    def set_line(self, x1: int, y1: int, x2: int, y2: int, color: str) -> tuple[bool, str]:
        idx = self._resolve_color(color)
        if idx is None:
            return False, f"Unknown color '{color}'"
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        while True:
            if self._grid.in_bounds(y, x) and not self._is_locked(y, x):
                self._grid.set(y, x, idx)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True, ""

    def flood_fill(self, x: int, y: int, color: str) -> tuple[bool, str]:
        idx = self._resolve_color(color)
        if idx is None:
            return False, f"Unknown color '{color}'"
        if not self._grid.in_bounds(y, x):
            return False, f"Position ({x},{y}) out of bounds"
        if self._is_locked(y, x):
            return False, f"Cell ({x},{y}) is a locked background cell"
        target = self._grid.get(y, x)
        if target == idx:
            return True, ""
        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited: set[tuple[int, int]] = set()
        while queue:
            cy, cx = queue.popleft()
            if (cy, cx) in visited:
                continue
            if not self._grid.in_bounds(cy, cx):
                continue
            if self._grid.get(cy, cx) != target:
                continue
            if self._is_locked(cy, cx):
                continue
            visited.add((cy, cx))
            self._grid.set(cy, cx, idx)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((cy + dy, cx + dx))
        return True, ""
