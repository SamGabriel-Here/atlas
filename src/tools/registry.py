"""Tool registry and dispatch.

Each tool module registers with @tool. The registry renders the API-facing
definitions and dispatches tool_use blocks to the matching Python function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class ApprovalFn(Protocol):
    def __call__(self, command: str, reason: str) -> bool: ...


@dataclass
class ToolContext:
    """Everything a tool needs that is not part of its declared input."""

    memory: Any = None
    approve: ApprovalFn | None = None
    cwd: str = "."


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


_REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, schema: dict[str, Any]) -> Callable:
    """Register a client-side tool.

    The description is the model's only guide to when this tool applies, so it
    states the trigger condition, not just the capability.
    """

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        _REGISTRY[name] = Tool(name=name, description=description, input_schema=schema, fn=fn)
        return fn

    return decorator


def client_tool_definitions() -> list[dict[str, Any]]:
    return [t.definition() for t in _REGISTRY.values()]


def dispatch(name: str, tool_input: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Run a tool. Returns (result_text, ok).

    Every failure is returned as text with ok=False rather than raised, so the
    model sees the error and can adapt instead of the loop dying.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        return f"Unknown tool: {name}", False
    try:
        return entry.fn(ctx=ctx, **(tool_input or {})), True
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}", False
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
        return f"{type(exc).__name__}: {exc}", False


def obj(**props: Any) -> dict[str, Any]:
    """Shorthand for a JSON Schema object with no extra properties."""
    required = [k for k, v in props.items() if v.pop("_required", False)]
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def s(desc: str, *, required: bool = False, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": desc, "_required": required, **extra}


def i(desc: str, *, required: bool = False, **extra: Any) -> dict[str, Any]:
    return {"type": "integer", "description": desc, "_required": required, **extra}


def b(desc: str, *, required: bool = False) -> dict[str, Any]:
    return {"type": "boolean", "description": desc, "_required": required}


def arr(desc: str, item_type: str = "string", *, required: bool = False) -> dict[str, Any]:
    return {
        "type": "array",
        "description": desc,
        "items": {"type": item_type},
        "_required": required,
    }
