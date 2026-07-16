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
    {
        "name": "set_rect",
        "description": "Fill a rectangle with a palette color.",
        "parameters": {
            "x1": "(int, required) Left column (inclusive).",
            "y1": "(int, required) Top row (inclusive).",
            "x2": "(int, required) Right column (inclusive).",
            "y2": "(int, required) Bottom row (inclusive).",
            "color": "(string, required) Color name from the active palette.",
            "filled": "(bool, optional, default true) If false, only draw the border.",
        },
    },
    {
        "name": "set_line",
        "description": "Draw a straight line between two points using Bresenham's algorithm.",
        "parameters": {
            "x1": "(int, required) Start column.",
            "y1": "(int, required) Start row.",
            "x2": "(int, required) End column.",
            "y2": "(int, required) End row.",
            "color": "(string, required) Color name from the active palette.",
        },
    },
    {
        "name": "flood_fill",
        "description": "Flood-fill from a seed point, replacing all connected same-color pixels.",
        "parameters": {
            "x": "(int, required) Seed column.",
            "y": "(int, required) Seed row.",
            "color": "(string, required) Replacement color name.",
        },
    },
]


def format_tools_for_prompt() -> str:
    lines = ["## Available Tools\n"]
    for spec in TOOL_SPECS:
        lines.append(f"### {spec['name']}")
        lines.append(f"{spec['description']}")
        lines.append("Parameters:")
        for k, v in spec["parameters"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def execute_tool(name: str, params: dict, edit_manager: EditManager) -> dict:
    """Dispatch a tool call to the EditManager. Returns {success, error}."""
    try:
        if name == "set_pixel":
            ok, err = edit_manager.set_pixel(
                x=int(params["x"]), y=int(params["y"]), color=str(params["color"])
            )
        elif name == "set_rect":
            ok, err = edit_manager.set_rect(
                x1=int(params["x1"]), y1=int(params["y1"]),
                x2=int(params["x2"]), y2=int(params["y2"]),
                color=str(params["color"]),
                filled=bool(params.get("filled", True)),
            )
        elif name == "set_line":
            ok, err = edit_manager.set_line(
                x1=int(params["x1"]), y1=int(params["y1"]),
                x2=int(params["x2"]), y2=int(params["y2"]),
                color=str(params["color"]),
            )
        elif name == "flood_fill":
            ok, err = edit_manager.flood_fill(
                x=int(params["x"]), y=int(params["y"]), color=str(params["color"])
            )
        else:
            return {"success": False, "error": f"Unknown tool '{name}'"}
        return {"success": ok, "error": err}
    except (KeyError, TypeError, ValueError) as e:
        return {"success": False, "error": f"Bad parameters for '{name}': {e}"}
