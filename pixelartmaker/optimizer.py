"""GreedyOptimizer — the agentic refinement loop."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time

from PIL import Image

from .edit_manager import EditManager
from .evaluator import CLIPEvaluator
from .grid import PixelGrid
from .renderer import render, render_to_bytes
from .tools import TOOL_SPECS, execute_tool, format_tools_for_prompt

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
        evaluator: CLIPEvaluator,
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

        self._client = self._init_client(provider, model, base_url)
        self.edit_manager: EditManager | None = None
        self._step_history: list[_StepRecord] = []
        self.current_score: float = float("-inf")
        self.step_count = 0
        self.accepted_frames: list[Image.Image] = []

    def _init_client(self, provider: str, model: str, base_url: str | None):
        if provider == "gemini":
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY env var required for gemini provider")
            return genai.Client(api_key=api_key)
        else:
            from openai import OpenAI
            if provider == "vllm":
                url = base_url or "http://localhost:8000/v1"
                return OpenAI(api_key="EMPTY", base_url=url)
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY env var required for openai provider")
                return OpenAI(api_key=api_key, base_url=base_url)

    def initialize(self, grid: PixelGrid, original_image: Image.Image | None = None) -> None:
        self.edit_manager = EditManager(grid)
        self.original_image = original_image
        rendered = render(self.edit_manager.grid)
        self.current_score = self._score(rendered)
        self.accepted_frames = [rendered.copy()]
        self.step_count = 0
        if self.verbose:
            print(f"Initial score: {self.current_score:.4f}")

    def _score(self, image: Image.Image, current_best: Image.Image | None = None) -> float:
        original = self.accepted_frames[0] if self.accepted_frames else None
        return self.evaluator.score(image, self.description, original=original, current_best=current_best)

    def _build_prompt(self, grid: PixelGrid) -> str:
        palette = grid.palette
        w, h = grid.width, grid.height

        accepted = [r for r in self._step_history if r.accepted]
        best_score = accepted[-1].score_after if accepted else self.current_score

        locked_count = int(grid.locked.sum())
        has_original = self.original_image is not None
        prompt = (
            f"You are a pixel art editor. Grid: {w}×{h}. Step {self.step_count + 1}.\n\n"
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
            f"- PRIORITY: match the original seed image pixel-for-pixel as closely as the palette allows. "
            f"Every edit must bring the grid closer to the original — do not invent details.\n"
            f"- Remove 'aura' pixels: stray pixels at the edge of the sprite that bleed the wrong color "
            f"into the background border (e.g. a faint purple halo around a dark outline). "
            f"Replace them with the correct outline or background-adjacent color from the palette.\n\n"
            f"## Output — respond with ONLY valid JSON, no other text:\n"
            f'{{"type":"STEP","rationale":"one sentence why","tool_calls":['
            f'{{"tool_name":"set_pixel","parameters":{{"x":5,"y":3,"color":"dark_purple"}}}}'
            f']}}\n'
            f'or {{"type":"STOP","rationale":"why you are done"}}\n\n'
            f"Up to {self.max_tool_calls} tool calls per step."
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

        if self.extra_instruction:
            prompt += f"\n\n## Extra Instruction\n{self.extra_instruction}"

        return prompt

    def _call_llm(self, prompt: str, image_bytes: bytes, original_bytes: bytes | None = None) -> str:
        if self.provider == "gemini":
            from google.genai import types
            contents = [prompt]
            if original_bytes:
                contents += [
                    "Image 1 — ORIGINAL TARGET:",
                    types.Part.from_bytes(data=original_bytes, mime_type="image/png"),
                    "Image 2 — CURRENT GRID (edit this):",
                ]
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            response = self._client.models.generate_content(model=self.model, contents=contents)
            return response.text
        else:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_url: dict = {"url": f"data:image/png;base64,{b64}"}
            if self.provider == "openai":
                image_url["detail"] = self.image_detail
            content = [{"type": "text", "text": prompt}]
            if original_bytes:
                orig_b64 = base64.b64encode(original_bytes).decode("utf-8")
                orig_url: dict = {"url": f"data:image/png;base64,{orig_b64}"}
                if self.provider == "openai":
                    orig_url["detail"] = self.image_detail
                content += [
                    {"type": "text", "text": "Image 1 — ORIGINAL TARGET:"},
                    {"type": "image_url", "image_url": orig_url},
                    {"type": "text", "text": "Image 2 — CURRENT GRID (edit this):"},
                ]
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
        raw = raw.strip()
        # Strip <think>...</think> blocks (Qwen reasoning models)
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
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

    def step(self, run_dir: str) -> bool:
        """Execute one optimization step. Returns True if STOP was requested."""
        self.step_count += 1
        grid = self.edit_manager.grid
        image_bytes = render_to_bytes(grid)
        prompt = self._build_prompt(grid)

        if self.debug:
            with open(os.path.join(run_dir, "debug.log"), "a") as f:
                f.write(f"\n{'='*60}\nSTEP {self.step_count}\n{'='*60}\n{prompt}\n")

        original_bytes = None
        if self.original_image is not None:
            buf = io.BytesIO()
            self.original_image.save(buf, format="PNG")
            original_bytes = buf.getvalue()

        t0 = time.time()
        try:
            raw = self._call_llm(prompt, image_bytes, original_bytes=original_bytes)
        except Exception as e:
            print(f"[ERROR] Step {self.step_count}: LLM call failed — {e}")
            return False

        elapsed = time.time() - t0

        if self.debug:
            with open(os.path.join(run_dir, "debug.log"), "a") as f:
                f.write(f"\n--- RESPONSE ---\n{raw}\n")

        parsed = self._parse_response(raw)
        if parsed is None:
            snippet = raw[-200:].replace("\n", " ") if raw else "(empty)"
            print(f"[ERROR] Step {self.step_count}: JSON parse failed. Tail: ...{snippet}")
            return False

        msg_type = parsed.get("type", "")

        if msg_type == "STOP":
            if self.verbose:
                print(f"  Step {self.step_count}: STOP — {parsed.get('rationale', '')}")
            return True

        if msg_type != "STEP":
            print(f"[ERROR] Step {self.step_count}: unknown message type '{msg_type}'. Full parsed: {parsed}")
            return False

        # Execute tool calls
        tool_calls = parsed.get("tool_calls", [])[:self.max_tool_calls]
        checkpoint = self.edit_manager.checkpoint()
        successes = 0
        for tc in tool_calls:
            result = execute_tool(tc.get("tool_name", ""), tc.get("parameters", {}), self.edit_manager)
            if result["success"]:
                successes += 1
            else:
                print(f"[ERROR] Tool '{tc.get('tool_name')}' failed: {result.get('error')}")

        new_grid = self.edit_manager.grid
        changed = new_grid.diff(grid)

        if changed == 0:
            self.edit_manager.rollback(checkpoint)
            if self.verbose:
                print(f"  Step {self.step_count}: no changes, rolled back")
            return False

        # Score new grid — pass current best frame so VLM can compare all three
        new_image = render(new_grid)
        current_best_frame = self.accepted_frames[-1] if self.accepted_frames else None
        new_score = self._score(new_image, current_best=current_best_frame)

        # Change penalty
        change_ratio = changed / grid.data.size
        penalty = 0.0
        if change_ratio > self.change_penalty_threshold:
            penalty = self.change_penalty_weight * (change_ratio - self.change_penalty_threshold)
        adjusted_score = new_score - penalty

        import random
        # VLM scorer returns >0.5 if candidate beats current best — use fixed threshold
        # CLIP scorer returns absolute similarity — compare against accumulated current_score
        vlm_mode = current_best_frame is not None and not hasattr(self.evaluator, '_model')
        threshold = 0.5 if vlm_mode else self.current_score
        accept = adjusted_score > threshold or (self.epsilon > 0 and random.random() < self.epsilon)

        rationale = parsed.get("rationale", "")
        self._step_history.append(_StepRecord(
            step=self.step_count,
            accepted=accept,
            score_before=self.current_score,
            score_after=new_score,
            rationale=rationale[:120],  # cap length so history stays compact
        ))

        if accept:
            self.current_score = new_score  # track raw score, penalty only for accept decision
            self.accepted_frames.append(new_image.copy())
            new_image.save(os.path.join(run_dir, f"step_{self.step_count:03d}_accepted.png"))
            if self.verbose:
                print(
                    f"  Step {self.step_count}: ACCEPTED  score {new_score:.4f} "
                    f"({successes}/{len(tool_calls)} tools ok, {changed} px changed, {elapsed:.1f}s)"
                )
        else:
            new_image.save(os.path.join(run_dir, f"step_{self.step_count:03d}_rejected.png"))
            self.edit_manager.rollback(checkpoint)
            if self.verbose:
                print(
                    f"  Step {self.step_count}: rejected  score {new_score:.4f} vs {self.current_score:.4f} "
                    f"({elapsed:.1f}s)"
                )

        return False

    def run(self, max_steps: int, run_dir: str) -> PixelGrid:
        for _ in range(max_steps):
            stop = self.step(run_dir)
            if stop:
                break
        return self.edit_manager.grid
