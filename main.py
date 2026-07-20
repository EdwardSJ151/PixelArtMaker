#!/usr/bin/env python3
"""PixelArtMaker — agentic pixel art refinement via vision LLM."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image

from pixelartmaker.evaluator import CLIPEvaluator, VLMEvaluator
from pixelartmaker.initializer import flatten_background, pixelate, select_sprite_size
from pixelartmaker.optimizer import GreedyOptimizer
from pixelartmaker.palette import Palette, best_n_from_palette
from pixelartmaker.renderer import render, save_gif
from pixelartmaker.utils import fg_bounding_box, make_client

load_dotenv()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_run_dir(description: str) -> str:
    slug = description[:30].replace(" ", "_").replace("/", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / "runs" / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _nes_color_count(grid_width: int, grid_height: int, nes_type: str) -> int:
    """Return usable color count for a NES sprite or background tile of given size."""
    if nes_type == "background":
        # BG tiles: one attribute byte per 16×16 block → always 3 usable colors
        return 3
    area = grid_width * grid_height
    if area <= 128:   # 8×8=64, 8×16=128
        return 3
    elif area <= 256: # 16×16=256
        return 4
    else:             # 16×32=512 and larger
        return 5


def main():
    parser = argparse.ArgumentParser(description="Refine pixel art with an agentic LLM loop.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    image_path  = cfg["image"]
    description = cfg["description"]
    extra       = cfg.get("extra_instruction", "")
    debug       = cfg.get("debug", False)

    llm_cfg  = cfg["llm"]
    # sprite: is the canonical section; fall back to palette: for old configs
    spr_cfg  = cfg.get("sprite", cfg.get("palette", {}))
    opt_cfg  = cfg.get("optimizer", {})
    clip_cfg = cfg.get("clip", {})
    out_cfg  = cfg.get("output", {})

    provider       = llm_cfg["provider"]
    model          = llm_cfg["model"]
    base_url       = llm_cfg.get("base_url")
    temperature    = llm_cfg.get("temperature", 0.7)
    max_tokens     = llm_cfg.get("max_tokens", 4096)
    image_detail   = llm_cfg.get("image_detail", "low")

    # ── sprite / palette config ────────────────────────────────────────────────
    system              = spr_cfg.get("system")
    nes_type            = spr_cfg.get("nes_type", "sprite")        # "sprite" | "background"
    sprite_size_str     = spr_cfg.get("size")                      # "16x16" or null
    size_presets_map    = spr_cfg.get("size_presets", {})
    palette_file        = spr_cfg.get("palette_file")
    palette_size        = spr_cfg.get("palette_size", spr_cfg.get("size", 16))
    sprite_only         = spr_cfg.get("sprite_only", False)
    second_white_threshold = spr_cfg.get("second_white_threshold")
    kmeans_n_init       = spr_cfg.get("kmeans_n_init", 10)
    kmeans_seed         = spr_cfg.get("kmeans_seed", 42)
    max_pixels          = spr_cfg.get("max_pixels", 50_000)

    # Preprocessing keys live in sprite: now; fall back to optimizer: for old configs
    remove_background  = spr_cfg.get("remove_background", opt_cfg.get("remove_background", False))
    bg_tolerance       = spr_cfg.get("bg_tolerance",      opt_cfg.get("bg_tolerance", 40))
    white_threshold    = spr_cfg.get("white_threshold",   opt_cfg.get("white_threshold", 240))
    resample           = spr_cfg.get("resample",          opt_cfg.get("resample", "nearest"))
    cell_size          = spr_cfg.get("cell_size",         opt_cfg.get("cell_size", 16))

    # ── optimizer config ───────────────────────────────────────────────────────
    max_steps                = opt_cfg.get("max_steps", 30)
    lock_background          = opt_cfg.get("lock_background", False)
    epsilon                  = opt_cfg.get("epsilon", 0.0)
    max_tool_calls           = opt_cfg.get("max_tool_calls", 4)
    change_penalty_threshold = opt_cfg.get("change_penalty_threshold", 0.4)
    change_penalty_weight    = opt_cfg.get("change_penalty_weight", 0.5)
    history_length           = opt_cfg.get("history_length", 0)

    harness_cfg       = cfg.get("harness", {})
    harness_mode      = harness_cfg.get("mode", "vlm")
    grid_ruler        = harness_cfg.get("grid_ruler", True)
    ascii_with_image  = harness_cfg.get("ascii_with_image", True)
    preview_highlight = harness_cfg.get("preview_highlight", "#FF4444")
    highlight_changes = harness_cfg.get("highlight_changes", False)

    clip_model      = clip_cfg.get("model", "hf-hub:timm/ViT-B-16-SigLIP")
    clip_pretrained = clip_cfg.get("pretrained", "")
    gif_duration_ms = out_cfg.get("gif_duration_ms", 200)

    seed_image = Image.open(image_path)
    print(f"Loaded seed image: {image_path} ({seed_image.size[0]}×{seed_image.size[1]})")

    # ── background removal ─────────────────────────────────────────────────────
    used_alpha_removal = False
    preflattened = None
    if remove_background:
        flat, bg_mask, used_alpha_removal = flatten_background(
            seed_image,
            tolerance=bg_tolerance,
            white_threshold=white_threshold,
            second_white_threshold=second_white_threshold,
        )
        bbox = fg_bounding_box(~bg_mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            palette_source = flat.crop((x1, y1, x2 + 1, y2 + 1))
            palette_bg_mask = bg_mask[y1:y2 + 1, x1:x2 + 1]
        else:
            palette_source = flat
            palette_bg_mask = bg_mask
        preflattened = (flat, bg_mask)
    else:
        palette_source = seed_image.convert("RGB")

    print(f"Provider: {provider} / Model: {model}")
    if system:
        print(f"System: {system}" + (f" / {nes_type}" if system == "nes" else ""))

    # ── sprite size ────────────────────────────────────────────────────────────
    if sprite_size_str:
        grid_width, grid_height = (int(x) for x in sprite_size_str.lower().split("x"))
        print(f"Sprite size: {grid_width}×{grid_height} (from config)")
    else:
        if system == "nes":
            presets = size_presets_map.get("nes", ["8x8", "8x16", "16x16", "16x32", "32x32"])
        elif system == "snes":
            presets = size_presets_map.get("snes", ["8x8", "16x16", "16x32", "32x32", "32x64", "64x64"])
        else:
            presets = size_presets_map.get("default", ["8x8", "16x16", "32x32", "48x48", "64x64"])

        print("Selecting sprite size via LLM...")
        init_client = make_client(provider, base_url)
        grid_width, grid_height = select_sprite_size(
            palette_source, description, init_client, model, provider, presets
        )

    # ── palette ────────────────────────────────────────────────────────────────
    if system == "nes":
        n_colors = _nes_color_count(grid_width, grid_height, nes_type)
        master_nes = Palette.from_system("nes")
        pixels_rgb = np.array(palette_source.convert("RGB"))
        palette = best_n_from_palette(pixels_rgb, master_nes, n_colors)
        label = f"NES {nes_type} palette: {n_colors} colors (from {len(master_nes.names)}-color master)"
        print(label)
        for name, hex_val in palette.named_colors.items():
            print(f"  {name:<6} {hex_val}")
    elif system == "snes":
        n_colors = 15
        print(f"Extracting SNES palette (k-means, {n_colors} colors)...")
        palette = Palette.extract_from_image(
            palette_source, n_colors=n_colors,
            kmeans_n_init=kmeans_n_init, kmeans_seed=kmeans_seed, max_pixels=max_pixels,
        )
        print("SNES palette:")
        for name, hex_val in palette.named_colors.items():
            print(f"  {name:<20} {hex_val}")
    elif palette_file:
        palette = Palette.from_file(palette_file)
        print(f"Palette from file: {palette_file} ({len(palette.names)} colors)")
    else:
        # Generic: k-means or fixed system (pico8/cga)
        if system:
            palette = Palette.from_system(system)
            print(f"Palette: {system} ({len(palette.names)} colors)")
        else:
            n_colors = int(palette_size) if str(palette_size).isdigit() else 16
            print(f"Extracting {n_colors}-color palette from seed image...")
            palette = Palette.extract_from_image(
                palette_source, n_colors=n_colors,
                kmeans_n_init=kmeans_n_init, kmeans_seed=kmeans_seed, max_pixels=max_pixels,
            )
            print("Palette:")
            for name, hex_val in palette.named_colors.items():
                print(f"  {name:<20} {hex_val}")

    # ── pixelate ───────────────────────────────────────────────────────────────
    print(f"\nPixelating to {grid_width}×{grid_height}...")
    grid = pixelate(
        seed_image, grid_width, grid_height, palette,
        resample=resample,
        preflattened=preflattened,
        lock_background=lock_background,
    )

    run_dir = make_run_dir(description)
    print(f"Run directory: {run_dir}")

    palette_source.save(os.path.join(run_dir, "cropped_original.png"))
    print(f"Cropped original saved → {run_dir}/cropped_original.png")

    initial_img = render(grid, cell_size=cell_size)
    initial_img.save(os.path.join(run_dir, "initial.png"))
    print(f"Initial grid saved → {run_dir}/initial.png")

    if sprite_only:
        print("sprite_only mode: done")
        return

    # ── evaluator + optimizer ──────────────────────────────────────────────────
    scorer_type = cfg.get("scorer", {}).get("type", "vlm")
    if scorer_type == "vlm":
        print(f"\nUsing VLM scorer ({model})...")
        evaluator = VLMEvaluator(provider=provider, model=model, base_url=base_url)
    else:
        print(f"\nLoading CLIP scorer ({clip_model})...")
        evaluator = CLIPEvaluator(model_name=clip_model, pretrained=clip_pretrained)

    optimizer = GreedyOptimizer(
        description=description,
        provider=provider,
        model=model,
        base_url=base_url,
        evaluator=evaluator,
        extra_instruction=extra,
        epsilon=epsilon,
        verbose=True,
        debug=debug,
        temperature=temperature,
        max_tokens=max_tokens,
        image_detail=image_detail,
        max_tool_calls=max_tool_calls,
        change_penalty_threshold=change_penalty_threshold,
        change_penalty_weight=change_penalty_weight,
        history_length=history_length,
        harness_mode=harness_mode,
        grid_ruler=grid_ruler,
        ascii_with_image=ascii_with_image,
        preview_highlight=preview_highlight,
        highlight_changes=highlight_changes,
    )
    optimizer.initialize(grid, original_image=palette_source, used_alpha_removal=used_alpha_removal)

    print(f"\nStarting optimization ({max_steps} steps)...")
    print("-" * 50)
    final_grid = optimizer.run(max_steps=max_steps, run_dir=run_dir)

    final_img = render(final_grid, cell_size=cell_size)
    final_img.save(os.path.join(run_dir, "final.png"))
    print(f"\nFinal grid saved → {run_dir}/final.png")

    if len(optimizer.accepted_frames) > 1:
        gif_path = os.path.join(run_dir, "progression.gif")
        save_gif(optimizer.accepted_frames, gif_path, duration_ms=gif_duration_ms)
        print(f"Progression GIF → {gif_path}")

    print(f"\nFinal score: {optimizer.current_score:.4f}")
    print(f"Accepted steps: {len(optimizer.accepted_frames) - 1} / {optimizer.step_count}")


if __name__ == "__main__":
    main()
