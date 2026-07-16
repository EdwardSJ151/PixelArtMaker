"""Tool definitions and registry for the LLM agent."""

from __future__ import annotations

from .edit_manager import EditManager


TOOL_SPECS = [
    {
        "name": "set_pixel",
        "description": "Set a single pixel to a palette color.",
        "parameters": {
            "x": "(int, required) Column, 0-indexed left to right.",
            "y": "(int, required) Row, 0-indexed top to bottom.",
            "color": "(string, required) Color name from the active palette.",
        },
    },
]


def _build_tools_prompt() -> str:
    lines = ["## Available Tools\n"]
    for spec in TOOL_SPECS:
        lines.append(f"### {spec['name']}")
        lines.append(f"{spec['description']}")
        lines.append("Parameters:")
        for k, v in spec["parameters"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    return "\n".join(lines)


_TOOLS_PROMPT = _build_tools_prompt()


def format_tools_for_prompt() -> str:
    return _TOOLS_PROMPT


def execute_tool(name: str, params: dict, edit_manager: EditManager) -> dict:
    """Dispatch a tool call to the EditManager. Returns {success, error}."""
    try:
        if name == "set_pixel":
            ok, err = edit_manager.set_pixel(
                x=int(params["x"]), y=int(params["y"]), color=str(params["color"])
            )
        else:
            return {"success": False, "error": f"Unknown tool '{name}'"}
        return {"success": ok, "error": err}
    except (KeyError, TypeError, ValueError) as e:
        return {"success": False, "error": f"Bad parameters for '{name}': {e}"}
