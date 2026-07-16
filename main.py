#!/usr/bin/env python3
"""PixelArtMaker — agentic pixel art generation via vision LLM + CLIP feedback."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from PIL import Image

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


def main():
    parser = argparse.ArgumentParser(description="Generate pixel art with an agentic LLM loop.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    image_path  = cfg["image"]
    description = cfg["description"]
    extra       = cfg.get("extra_instruction", "")
    debug       = cfg.get("debug", False)

    llm_cfg  = cfg["llm"]
    pal_cfg  = cfg["palette"]
    opt_cfg  = cfg["optimizer"]
    clip_cfg = cfg.get("clip", {})
    out_cfg  = cfg.get("output", {})

    provider       = llm_cfg["provider"]
    model          = llm_cfg["model"]
    base_url       = llm_cfg.get("base_url")
    temperature    = llm_cfg.get("temperature", 0.7)
    max_tokens     = llm_cfg.get("max_tokens", 4096)
    image_detail   = llm_cfg.get("image_detail", "low")

    pal_size        = pal_cfg["size"]
    system          = pal_cfg.get("system")
    kmeans_n_init   = pal_cfg.get("kmeans_n_init", 10)
    kmeans_seed     = pal_cfg.get("kmeans_seed", 42)
    max_pixels      = pal_cfg.get("max_pixels", 50_000)

    resample                 = opt_cfg.get("resample", "nearest")
    remove_background        = opt_cfg.get("remove_background", False)
    bg_tolerance             = opt_cfg.get("bg_tolerance", 40)
    max_steps                = opt_cfg["max_steps"]
    grid_size                = opt_cfg.get("grid_size")
    grid_presets             = opt_cfg.get("grid_presets", [8, 16, 32, 48, 64])
    epsilon                  = opt_cfg.get("epsilon", 0.0)
    cell_size                = opt_cfg.get("cell_size", 16)
    max_tool_calls           = opt_cfg.get("max_tool_calls", 30)
    change_penalty_threshold = opt_cfg.get("change_penalty_threshold", 0.4)
    change_penalty_weight    = opt_cfg.get("change_penalty_weight", 0.5)
    history_length           = opt_cfg.get("history_length", 0)

    clip_model     = clip_cfg.get("model", "ViT-B-32")
    clip_pretrained = clip_cfg.get("pretrained", "openai")

    gif_duration_ms = out_cfg.get("gif_duration_ms", 200)

    from pixelartmaker.evaluator import CLIPEvaluator
    from pixelartmaker.initializer import pixelate, select_grid_size, _flatten_background
    from pixelartmaker.optimizer import GreedyOptimizer
    from pixelartmaker.palette import Palette
    from pixelartmaker.renderer import render, save_gif

    seed_image = Image.open(image_path).convert("RGB")
    print(f"Loaded seed image: {image_path} ({seed_image.size[0]}×{seed_image.size[1]})")

    # Flatten background before palette extraction so halo colors don't claim palette slots
    palette_source = _flatten_background(seed_image, tolerance=bg_tolerance) if remove_background else seed_image

    print(f"Provider: {provider} / Model: {model}")

    if grid_size:
        print(f"Grid size: {grid_size}×{grid_size} (from config)")
    else:
        print("Selecting grid size via LLM...")
        if provider == "gemini":
            from google import genai
            init_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        elif provider == "vllm":
            from openai import OpenAI
            init_client = OpenAI(api_key="EMPTY", base_url=base_url or "http://localhost:8000/v1")
        else:
            from openai import OpenAI
            init_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        grid_size = select_grid_size(seed_image, description, init_client, model, provider, grid_presets)

    if system:
        palette = Palette.from_system(system)
        print(f"Palette: {system} system ({len(palette.names)} colors)")
    else:
        print(f"Extracting {pal_size}-color palette from seed image...")
        palette = Palette.extract_from_image(
            palette_source, n_colors=pal_size,
            kmeans_n_init=kmeans_n_init, kmeans_seed=kmeans_seed, max_pixels=max_pixels,
        )
        print("Palette:")
        for name, hex_val in palette.named_colors.items():
            print(f"  {name:<20} {hex_val}")

    print(f"\nPixelating to {grid_size}×{grid_size}...")
    grid = pixelate(seed_image, grid_size, palette, resample=resample,
                    remove_background=remove_background, bg_tolerance=bg_tolerance)

    run_dir = make_run_dir(description)
    print(f"Run directory: {run_dir}")

    initial_img = render(grid, cell_size=cell_size)
    initial_img.save(os.path.join(run_dir, "initial.png"))
    print(f"Initial grid saved → {run_dir}/initial.png")

    print(f"\nLoading scoring model ({clip_model})...")
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
    )
    optimizer.initialize(grid)

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

    print(f"\nFinal CLIP score: {optimizer.current_score:.4f}")
    print(f"Accepted steps: {len(optimizer.accepted_frames) - 1} / {optimizer.step_count}")


if __name__ == "__main__":
    main()
