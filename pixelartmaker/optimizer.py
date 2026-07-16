"""GreedyOptimizer — the agentic refinement loop."""

from __future__ import annotations

import json
import os
import random
import re
import time

import numpy as np
from PIL import Image

from .edit_manager import EditManager
from .evaluator import CLIPEvaluator, VLMEvaluator
from .grid import PixelGrid
from .renderer import render, render_to_bytes, render_with_ruler, render_ascii, render_preview
from .tools import execute_tool, format_tools_for_prompt
from .utils import make_client, strip_think_tags, img_to_bytes

class _StepRecord:
    __slots__ = ("step", "accepted", "score_before", "score_after", "rationale")

    def __init__(self, step: int, accepted: bool, score_before: float, score_after: float, rationale: str):
        self.step = step
        self.accepted = accepted
        self.score_before = score_before
        self.score_after = score_after
        self.rationale = rationale

    def __str__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (
            f"Step {self.step} ({status}, {self.score_before:.4f} → {self.score_after:.4f}): "
            f"{self.rationale}"
        )


class GreedyOptimizer:
    def __init__(
        self,
        description: str,
        provider: str,
        model: str,
        base_url: str | None,
        evaluator: CLIPEvaluator | VLMEvaluator,
        extra_instruction: str = "",
        epsilon: float = 0.0,
        verbose: bool = True,
        debug: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        image_detail: str = "low",
        max_tool_calls: int = 30,
        change_penalty_threshold: float = 0.4,
        change_penalty_weight: float = 0.5,
        history_length: int = 0,
        harness_mode: str = "vlm",
        grid_ruler: bool = True,
        ascii_with_image: bool = True,
        preview_highlight: str = "#FF4444",
    ):
        self.description = description
        self.provider = provider
        self.model = model
        self.evaluator = evaluator
        self.extra_instruction = extra_instruction
        self.epsilon = epsilon
        self.verbose = verbose
        self.debug = debug
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_detail = image_detail
        self.max_tool_calls = max_tool_calls
        self.change_penalty_threshold = change_penalty_threshold
        self.change_penalty_weight = change_penalty_weight
        self.history_length = history_length
        self.harness_mode = harness_mode
        self.grid_ruler = grid_ruler
        self.ascii_with_image = ascii_with_image
        self.preview_highlight = preview_highlight

        self._client = make_client(provider, base_url)
        self.edit_manager: EditManager | None = None
        self._step_history: list[_StepRecord] = []
        self.current_score: float = float("-inf")
        self.step_count = 0
        self.accepted_frames: list[Image.Image] = []

    def initialize(self, grid: PixelGrid, original_image: Image.Image | None = None, used_alpha_removal: bool = False) -> None:
        self.edit_manager = EditManager(grid)
        self.original_image = original_image
        self._used_alpha_removal = used_alpha_removal
        rendered = render(self.edit_manager.grid)
        self.current_score = self._score(rendered)
        self.accepted_frames = [rendered.copy()]
        self.step_count = 0
        if self.verbose:
            print(f"Initial score: {self.current_score:.4f}")

    def _score(self, image: Image.Image, current_best: Image.Image | None = None) -> float:
        original = self.accepted_frames[0] if self.accepted_frames else None
        return self.evaluator.score(image, self.description, original=original, current_best=current_best)

    def _build_prompt(self, grid: PixelGrid, ascii_grid: str | None = None) -> str:
        palette = grid.palette
        w, h = grid.width, grid.height

        accepted = [r for r in self._step_history if r.accepted]
        best_score = accepted[-1].score_after if accepted else self.current_score

        locked_count = int(grid.locked.sum())
        has_original = self.original_image is not None
        ruler_note = (
            " The image has coordinate rulers — numbers along the top are column (x) indices, "
            "numbers along the left are row (y) indices. Use these to reference exact cell positions."
            if self.harness_mode in ("vlm", "ascii+vlm") and self.grid_ruler else ""
        )
        prompt = (
            f"You are a pixel art editor. Grid: {w}×{h}. Step {self.step_count + 1}.{ruler_note}\n\n"
            + (
                f"You are given TWO images:\n"
                f"  • Image 1 — ORIGINAL TARGET: the source image you must reproduce\n"
                f"  • Image 2 — CURRENT GRID: the pixelated canvas you are editing\n\n"
                f"YOUR ONLY TASK: make Image 2 match Image 1 as closely as possible "
                f"given the palette and grid resolution constraints. "
                f"Study Image 1 carefully — shapes, colors, outlines, proportions — then edit Image 2.\n"
                if has_original else
                f"YOUR ONLY TASK: make this grid match the original seed image as closely as possible "
                f"given the palette and grid resolution constraints.\n"
            )
            + f'Subject: "{self.description}"\n\n'
            f"Do NOT interpret this description creatively. Do NOT add features, emotions, expressions, "
            f"or artistic elements that are not visible in the original. You are correcting a pixelated "
            f"reproduction, not designing something new.\n\n"
            f"Similarity score: {self.current_score:.4f} (best: {best_score:.4f}) — higher = closer to original.\n\n"
            f"Palette — ONLY these exact color names:\n"
            f"{palette.format_for_prompt()}\n\n"
            f"{format_tools_for_prompt()}\n"
            f"## Hard rules\n"
            f"- {locked_count} cells are locked background — any edit to them will be rejected.\n"
            f"- Small targeted edits only. Changing >{int(self.change_penalty_threshold*100)}% of pixels is penalized.\n"
            f"- Do NOT repeat rejected approaches.\n"
            f"- IMPORTANT: if a step is rejected, it is automatically rolled back — you always start "
            f"from the last accepted state. Never output STOP because of a rejection; just try a "
            f"different smaller edit.\n"
            f"- PRIORITY: match the original seed image pixel-for-pixel as closely as the palette allows. "
            f"Every edit must bring the grid closer to the original — do not invent details.\n"
            + (
            f"- Remove 'aura' pixels: the original image had semi-transparent edge pixels (from "
            f"anti-aliasing) that may appear as a faint halo around the sprite. Find stray off-color "
            f"pixels at sprite edges and replace them with the correct outline or nearest solid palette color.\n"
            if self._used_alpha_removal else ""
            )
            + "\n"
            f"## Output — respond with ONLY valid JSON, no other text:\n"
            f'{{"type":"STEP","rationale":"one sentence why","tool_calls":['
            f'{{"tool_name":"set_pixel","parameters":{{"x":5,"y":3,"color":"dark_purple"}}}}'
            f']}}\n'
            f'or {{"type":"STOP","rationale":"why you are done"}}\n\n'
            f"IMPORTANT: include exactly ONE tool call. It will be evaluated immediately — "
            f"if it improves the image it is committed and the step ends. "
            f"If not, it is rolled back and you will be asked again (up to {self.max_tool_calls} attempts). "
            f"Only output STOP if the image already matches the target well."
        )

        if self.history_length > 0 and self._step_history:
            recent = self._step_history[-self.history_length:]
            accepted_lines = [f"  ✓ Step {r.step} (+{r.score_after - r.score_before:+.4f}): {r.rationale}" for r in recent if r.accepted]
            rejected_lines = [f"  ✗ Step {r.step} ({r.score_after - r.score_before:+.4f}): {r.rationale}" for r in recent if not r.accepted]
            history = ""
            if accepted_lines:
                history += "Accepted (these worked):\n" + "\n".join(accepted_lines) + "\n"
            if rejected_lines:
                history += "Rejected (score dropped — avoid repeating these):\n" + "\n".join(rejected_lines)
            if history:
                prompt += f"\n\n## Recent history\n{history.strip()}"

        if ascii_grid:
            prompt += f"\n\n## ASCII Grid (current state)\n{ascii_grid}"

        if self.extra_instruction:
            prompt += f"\n\n## Extra Instruction\n{self.extra_instruction}"

        return prompt

    def _call_llm(self, prompt: str, image_bytes: bytes | None, original_bytes: bytes | None = None) -> str:
        if self.provider == "gemini":
            from google.genai import types
            contents = [prompt]
            if original_bytes:
                contents += [
                    "Image 1 — ORIGINAL TARGET:",
                    types.Part.from_bytes(data=original_bytes, mime_type="image/png"),
                    "Image 2 — CURRENT GRID (edit this):",
                ]
            if image_bytes is not None:
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            response = self._client.models.generate_content(model=self.model, contents=contents)
            return response.text
        else:
            import base64
            content = [{"type": "text", "text": prompt}]
            if original_bytes:
                orig_url: dict = {"url": f"data:image/png;base64,{base64.b64encode(original_bytes).decode()}"}
                if self.provider == "openai":
                    orig_url["detail"] = self.image_detail
                content += [
                    {"type": "text", "text": "Image 1 — ORIGINAL TARGET:"},
                    {"type": "image_url", "image_url": orig_url},
                    {"type": "text", "text": "Image 2 — CURRENT GRID (edit this):"},
                ]
            if image_bytes is not None:
                image_url: dict = {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"}
                if self.provider == "openai":
                    image_url["detail"] = self.image_detail
                content.append({"type": "image_url", "image_url": image_url})
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> dict | None:
        """Extract JSON from LLM response, handling think-tags and markdown fences."""
        raw = strip_think_tags(raw.strip())
        # Strip markdown code fences
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            raw = match.group(1).strip()
        # Find first { ... } block
        try:
            start = raw.index("{")
            return json.loads(raw[start:])
        except (ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _diff_cells(old_grid: PixelGrid, new_grid: PixelGrid) -> list[tuple[int, int]]:
        """Return list of (x, y) cell positions that differ between two grids."""
        ys, xs = np.where(old_grid.data != new_grid.data)
        return list(zip(xs.tolist(), ys.tolist()))

    def _preview_confirm(self, tc: dict, grid: PixelGrid, run_dir: str) -> bool:
        """Apply tc to a scratch copy, show highlighted preview, ask model CONFIRM/REVISE.

        Returns True if the model confirms (or gives an ambiguous response).
        """
        scratch_em = EditManager(grid.copy())
        result = execute_tool(tc.get("tool_name", ""), tc.get("parameters", {}), scratch_em)
        if not result["success"]:
            return True  # let the normal execution path handle the failure

        affected = self._diff_cells(grid, scratch_em.grid)
        if not affected:
            return True

        preview_img = render_preview(scratch_em.grid, affected, highlight_color=self.preview_highlight)
        confirm_prompt = (
            "This is a preview of your proposed edit. Highlighted cells (red border) show what will change.\n"
            "Reply with exactly one word: CONFIRM (to apply the edit) or REVISE (to try a different edit)."
        )
        try:
            raw = self._call_llm(confirm_prompt, img_to_bytes(preview_img))
        except Exception as e:  # noqa: BLE001
            with open(os.path.join(run_dir, "generator.log"), "a") as f:
                f.write(f"[preview confirm ERROR] {e}\n")
            return True  # default to confirm on error

        with open(os.path.join(run_dir, "generator.log"), "a") as f:
            f.write(f"[preview confirm] {raw.strip()[:80]}\n")

        return "REVISE" not in strip_think_tags(raw).upper()

    def step(self, run_dir: str) -> bool:
        """Execute one optimization step.

        Each attempt asks the LLM for a single tool call, evaluates it immediately,
        and commits if it improves the score. The step ends as soon as one attempt
        is accepted, or after max_tool_calls failed attempts.

        Returns True if STOP was requested.
        """
        self.step_count += 1

        original_bytes = img_to_bytes(self.original_image) if self.original_image is not None else None

        vlm_mode = isinstance(self.evaluator, VLMEvaluator)

        for attempt in range(1, self.max_tool_calls + 1):
            grid = self.edit_manager.grid

            # Build image and prompt according to harness mode
            mode = self.harness_mode
            ascii_str: str | None = None
            if mode in ("ascii", "ascii+vlm"):
                ascii_str, _ = render_ascii(grid)

            if mode == "vlm":
                rendered_img = render_with_ruler(grid) if self.grid_ruler else render(grid)
                image_bytes = img_to_bytes(rendered_img)
            elif mode == "ascii":
                rendered_img = render(grid) if self.ascii_with_image else None
                image_bytes = img_to_bytes(rendered_img) if rendered_img is not None else None
            elif mode == "ascii+vlm":
                rendered_img = render_with_ruler(grid) if self.grid_ruler else render(grid)
                image_bytes = img_to_bytes(rendered_img)
            else:  # preview or unknown — default to plain render
                rendered_img = render(grid)
                image_bytes = img_to_bytes(rendered_img)

            prompt = self._build_prompt(grid, ascii_grid=ascii_str)

            if self.debug:
                with open(os.path.join(run_dir, "debug.log"), "a") as f:
                    f.write(f"\n{'='*60}\nSTEP {self.step_count} attempt {attempt}\n{'='*60}\n{prompt}\n")

            t0 = time.time()
            try:
                raw = self._call_llm(prompt, image_bytes, original_bytes=original_bytes)
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Step {self.step_count} attempt {attempt}: LLM call failed — {e}")
                with open(os.path.join(run_dir, "generator.log"), "a") as f:
                    f.write(f"\n[Step {self.step_count} attempt {attempt}] LLM ERROR: {e}\n")
                continue

            elapsed = time.time() - t0

            with open(os.path.join(run_dir, "generator.log"), "a") as f:
                f.write(f"\n{'='*60}\nStep {self.step_count} | attempt {attempt} | {elapsed:.1f}s\n{'='*60}\n{raw}\n")

            if self.debug:
                with open(os.path.join(run_dir, "debug.log"), "a") as f:
                    f.write(f"\n--- RESPONSE ---\n{raw}\n")

            parsed = self._parse_response(raw)
            if parsed is None:
                snippet = raw[-200:].replace("\n", " ") if raw else "(empty)"
                print(f"[ERROR] Step {self.step_count} attempt {attempt}: JSON parse failed. Tail: ...{snippet}")
                continue

            msg_type = parsed.get("type", "")

            if msg_type == "STOP":
                if self.verbose:
                    print(f"  Step {self.step_count}: STOP — {parsed.get('rationale', '')}")
                return True

            if msg_type != "STEP":
                print(f"[ERROR] Step {self.step_count} attempt {attempt}: unknown type '{msg_type}'")
                continue

            # Take only the first tool call
            tool_calls = parsed.get("tool_calls", [])
            if not tool_calls:
                print(f"[WARN] Step {self.step_count} attempt {attempt}: no tool calls in response")
                continue

            tc = tool_calls[0]

            # Preview mode: confirm with model before applying
            if self.harness_mode == "preview":
                confirmed = self._preview_confirm(tc, grid, run_dir)
                if not confirmed:
                    with open(os.path.join(run_dir, "generator.log"), "a") as f:
                        f.write(f"→ PREVIEW REVISE (attempt {attempt})\n")
                    if self.verbose:
                        print(f"  Step {self.step_count} attempt {attempt}: preview REVISE")
                    continue

            checkpoint = self.edit_manager.checkpoint()
            result = execute_tool(tc.get("tool_name", ""), tc.get("parameters", {}), self.edit_manager)
            if not result["success"]:
                print(f"[ERROR] Step {self.step_count} attempt {attempt}: tool failed — {result.get('error')}")
                self.edit_manager.rollback(checkpoint)
                continue

            new_grid = self.edit_manager.grid
            changed = new_grid.diff(grid)
            if changed == 0:
                self.edit_manager.rollback(checkpoint)
                continue

            new_image = render(new_grid)
            current_best_frame = self.accepted_frames[-1] if self.accepted_frames else None
            new_score = self._score(new_image, current_best=current_best_frame)

            change_ratio = changed / grid.data.size
            penalty = 0.0
            if change_ratio > self.change_penalty_threshold:
                penalty = self.change_penalty_weight * (change_ratio - self.change_penalty_threshold)
            adjusted_score = new_score - penalty

            threshold = 0.5 if vlm_mode else self.current_score
            accept = adjusted_score > threshold or (self.epsilon > 0 and random.random() < self.epsilon)

            rationale = parsed.get("rationale", "")
            self._step_history.append(_StepRecord(
                step=self.step_count,
                accepted=accept,
                score_before=self.current_score,
                score_after=new_score,
                rationale=rationale[:120],
            ))

            if accept:
                self.current_score = new_score
                self.accepted_frames.append(new_image.copy())
                new_image.save(os.path.join(run_dir, f"step_{self.step_count:03d}_accepted.png"))
                with open(os.path.join(run_dir, "generator.log"), "a") as f:
                    f.write(f"→ ACCEPTED  score {new_score:.4f}  changed={changed}px\n")
                if self.verbose:
                    print(
                        f"  Step {self.step_count}: ACCEPTED  score {new_score:.4f} "
                        f"({changed} px changed, attempt {attempt}/{self.max_tool_calls}, {elapsed:.1f}s)"
                    )
                return False
            else:
                new_image.save(os.path.join(run_dir, f"step_{self.step_count:03d}_a{attempt}_rejected.png"))
                self.edit_manager.rollback(checkpoint)
                with open(os.path.join(run_dir, "generator.log"), "a") as f:
                    f.write(f"→ REJECTED  score {new_score:.4f} vs {self.current_score:.4f}\n")
                if self.verbose:
                    print(
                        f"  Step {self.step_count} attempt {attempt}: rejected  "
                        f"score {new_score:.4f} vs {self.current_score:.4f} ({elapsed:.1f}s)"
                    )

        if self.verbose:
            print(f"  Step {self.step_count}: all {self.max_tool_calls} attempts rejected")
        return False

    def run(self, max_steps: int, run_dir: str) -> PixelGrid:
        for _ in range(max_steps):
            stop = self.step(run_dir)
            if stop:
                break
        return self.edit_manager.grid
