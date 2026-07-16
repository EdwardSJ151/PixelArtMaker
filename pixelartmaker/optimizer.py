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

        self._client = self._init_client(provider, model, base_url)
        self.edit_manager: EditManager | None = None
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

    def initialize(self, grid: PixelGrid) -> None:
        self.edit_manager = EditManager(grid)
        rendered = render(self.edit_manager.grid)
        self.current_score = self._score(rendered)
        self.accepted_frames = [rendered.copy()]
        self.step_count = 0
        if self.verbose:
            print(f"Initial CLIP score: {self.current_score:.4f}")

    def _score(self, image: Image.Image) -> float:
        return self.evaluator.score(image, self.description)

    def _build_prompt(self, grid: PixelGrid) -> str:
        palette = grid.palette
        w, h = grid.width, grid.height

        prompt = (
            f"You are a pixel art editor working on a {w}×{h} grid.\n"
            f'Target description: "{self.description}"\n\n'
            f"The rendered image of the current pixel art grid is attached.\n\n"
            f"Active palette — use ONLY these color names in tool calls:\n"
            f"{palette.format_for_prompt()}\n\n"
            f"CLIP similarity score: {self.current_score:.4f}  (goal: maximize — higher means closer to description)\n\n"
            f"{format_tools_for_prompt()}\n"
            f"## Instructions\n"
            f"1. Look at the rendered image and compare it to the description.\n"
            f"2. Identify pixels or regions that don't match the description.\n"
            f"3. Make targeted tool calls to improve those areas.\n"
            f"4. Prefer small, precise edits over large rewrites.\n\n"
            f"## Response Format\n"
            f"Respond with ONLY a JSON object:\n\n"
            f"### STEP — make edits:\n"
            f'{{"type":"STEP","rationale":"your reasoning","tool_calls":['
            f'{{"tool_name":"set_pixel","parameters":{{"x":5,"y":3,"color":"dark_purple"}}}}'
            f']}}\n\n'
            f"### STOP — when satisfied:\n"
            f'{{"type":"STOP","rationale":"why you are done"}}\n\n'
            f"Note: You may use up to {self.max_tool_calls} tool calls per step."
        )

        if self.extra_instruction:
            prompt += f"\n\n## Extra Instruction\n{self.extra_instruction}"

        return prompt

    def _call_llm(self, prompt: str, image_bytes: bytes) -> str:
        if self.provider == "gemini":
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
            )
            return response.text
        else:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_url: dict = {"url": f"data:image/png;base64,{b64}"}
            if self.provider == "openai":
                image_url["detail"] = self.image_detail
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url},
            ]
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> dict | None:
        """Extract JSON from LLM response."""
        raw = raw.strip()
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

        t0 = time.time()
        try:
            raw = self._call_llm(prompt, image_bytes)
        except Exception as e:
            print(f"  Step {self.step_count}: LLM error — {e}")
            return False

        elapsed = time.time() - t0

        if self.debug:
            with open(os.path.join(run_dir, "debug.log"), "a") as f:
                f.write(f"\n--- RESPONSE ---\n{raw}\n")

        parsed = self._parse_response(raw)
        if parsed is None:
            print(f"  Step {self.step_count}: parse error")
            return False

        msg_type = parsed.get("type", "")

        if msg_type == "STOP":
            if self.verbose:
                print(f"  Step {self.step_count}: STOP — {parsed.get('rationale', '')}")
            return True

        if msg_type != "STEP":
            print(f"  Step {self.step_count}: unknown message type '{msg_type}'")
            return False

        # Execute tool calls
        tool_calls = parsed.get("tool_calls", [])[:self.max_tool_calls]
        checkpoint = self.edit_manager.checkpoint()
        successes = 0
        for tc in tool_calls:
            result = execute_tool(tc.get("tool_name", ""), tc.get("parameters", {}), self.edit_manager)
            if result["success"]:
                successes += 1

        new_grid = self.edit_manager.grid
        changed = new_grid.diff(grid)

        if changed == 0:
            self.edit_manager.rollback(checkpoint)
            if self.verbose:
                print(f"  Step {self.step_count}: no changes, rolled back")
            return False

        # Score new grid
        new_image = render(new_grid)
        new_score = self._score(new_image)

        # Change penalty
        change_ratio = changed / grid.data.size
        penalty = 0.0
        if change_ratio > self.change_penalty_threshold:
            penalty = self.change_penalty_weight * (change_ratio - self.change_penalty_threshold)
        adjusted_score = new_score - penalty

        import random
        accept = adjusted_score > self.current_score or (self.epsilon > 0 and random.random() < self.epsilon)

        if accept:
            self.current_score = new_score  # track raw score, penalty only for accept decision
            self.accepted_frames.append(new_image.copy())
            new_image.save(os.path.join(run_dir, f"step_{self.step_count:03d}.png"))
            if self.verbose:
                print(
                    f"  Step {self.step_count}: ACCEPTED  score {new_score:.4f} "
                    f"({successes}/{len(tool_calls)} tools ok, {changed} px changed, {elapsed:.1f}s)"
                )
        else:
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
