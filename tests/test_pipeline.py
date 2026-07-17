"""Pipeline unit tests — no LLM or ML required."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from pixelartmaker.grid import PixelGrid
from pixelartmaker.palette import Palette
from pixelartmaker.renderer import render, render_with_ruler
from pixelartmaker.edit_manager import EditManager
from pixelartmaker.optimizer import GreedyOptimizer


# ── shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def palette():
    return Palette.from_system("pico8")


@pytest.fixture
def grid(palette):
    data = np.zeros((8, 8), dtype=np.int32)
    data[2:6, 2:6] = 3  # center square is color index 3
    return PixelGrid(data, palette)


# ── test 1: ruler adds correct margins ────────────────────────────────────────

def test_render_with_ruler_dimensions(grid):
    """render_with_ruler should add exactly 24px on left and top."""
    cell_size = 16
    base = render(grid, cell_size)
    ruled = render_with_ruler(grid, cell_size)

    assert ruled.size == (base.width + 24, base.height + 24)
    # Pixel at (0,0) is in the margin — should be the light gray fill, not a grid cell
    r, g, b = ruled.getpixel((0, 0))
    # grid cell 0,0 is black (#000000); margin is (240,240,240) — they must differ
    cell_color = base.getpixel((0, 0))
    assert (r, g, b) != cell_color, "Top-left margin pixel should not match grid cell color"


# ── test 2: set_pixel diff detects the changed cell ───────────────────────────

def test_set_pixel_diff_detection(grid, palette):
    """After set_pixel, diff against checkpoint must return exactly 1 changed cell."""
    em = EditManager(grid)
    checkpoint = em.checkpoint()

    # cell (3,3) starts as index 3; change it to index 0 (black)
    original_idx = grid.data[3, 3]
    target_color = palette.names[0]  # index 0
    assert original_idx != 0, "precondition: (3,3) is not already color 0"

    ok, err = em.set_pixel(x=3, y=3, color=target_color)
    assert ok, f"set_pixel failed: {err}"

    changed = em.grid.diff(checkpoint)
    assert changed == 1, f"expected 1 changed cell, got {changed}"

    # diff_cells gives us (x, y) pairs
    diff_cells = GreedyOptimizer._diff_cells(checkpoint, em.grid)
    assert diff_cells == [(3, 3)], f"wrong cell reported changed: {diff_cells}"


# ── test 3: step accepts edit when mock evaluator approves ────────────────────

class _AlwaysAcceptEvaluator:
    """Stub evaluator that always returns 0.9 (candidate clearly better)."""
    def score(self, image, description, original=None, current_best=None, **kwargs):
        if original is None or current_best is None:
            return 0.5
        return 0.9


def test_step_saves_correct_image(grid, palette):
    """When the mock evaluator accepts, the saved PNG must contain the changed pixel color."""
    original_image = render(grid)  # use initial render as stand-in for source

    evaluator = _AlwaysAcceptEvaluator()
    with patch("pixelartmaker.optimizer.make_client", return_value=None):
        optimizer = GreedyOptimizer(
            description="test",
            provider="openai",
            model="gpt-4o",
            base_url=None,
            evaluator=evaluator,
            max_tool_calls=1,
        )
    optimizer.initialize(grid, original_image=original_image)

    # Patch _call_llm to return a set_pixel proposal targeting cell (0,0) → color index 5
    target_color = palette.names[5]
    fake_response = json.dumps({
        "type": "STEP",
        "rationale": "test edit",
        "tool_calls": [{"tool_name": "set_pixel", "parameters": {"x": 0, "y": 0, "color": target_color}}],
    })

    with tempfile.TemporaryDirectory() as run_dir:
        with patch.object(optimizer, "_call_llm", return_value=fake_response):
            stop = optimizer.step(run_dir)

        assert not stop, "step should not STOP"
        assert optimizer.current_score == 0.9, "score should be updated to evaluator's value"
        assert len(optimizer.accepted_frames) == 2, "accepted_frames should grow by 1"

        saved_path = os.path.join(run_dir, "step_001_accepted.png")
        assert os.path.exists(saved_path), "accepted PNG not saved"

        saved = Image.open(saved_path)
        cell_size = 16
        # Sample the center of cell (0,0) in the saved image
        px = saved.getpixel((cell_size // 2, cell_size // 2))

        expected_hex = palette.hex_of(palette.index_of(target_color))
        expected_rgb = (
            int(expected_hex[1:3], 16),
            int(expected_hex[3:5], 16),
            int(expected_hex[5:7], 16),
        )
        assert px == expected_rgb, (
            f"saved image cell (0,0) is {px}, expected {expected_rgb} ({target_color})"
        )
