"""Memory tools: the agent's own handle on its long-term store.

Recall also happens automatically each turn; these let the model store new
facts and dig deeper on demand.
"""

from __future__ import annotations

from src.tools.registry import ToolContext, arr, i, obj, s, tool


@tool(
    "remember",
    "Save a durable fact to long-term memory so it is available in future sessions. "
    "Call this when the user tells you something about themselves, their preferences, "
    "their projects, or their environment that will still be true next week - not for "
    "details that only matter inside the current conversation.",
    obj(
        text=s(
            "The fact, written as a self-contained sentence that will make sense with no "
            "surrounding conversation.",
            required=True,
        ),
        tags=arr("Short topic tags for filtering later, e.g. ['preferences', 'python']."),
    ),
)
def remember(ctx: ToolContext, text: str, tags: list[str] | None = None) -> str:
    if ctx.memory is None or not getattr(ctx.memory, "enabled", False):
        return "Long-term memory is disabled, so nothing was saved."
    mem_id = ctx.memory.remember(text, tags or [])
    return f"Saved to long-term memory (id {mem_id})."


@tool(
    "recall",
    "Search long-term memory for anything relevant to a query. The most relevant "
    "memories are already injected each turn, so use this when you need to dig further "
    "back or search a different topic than the one at hand.",
    obj(
        query=s("What to search for, in natural language.", required=True),
        limit=i("How many memories to return. Defaults to 5."),
    ),
)
def recall(ctx: ToolContext, query: str, limit: int = 5) -> str:
    if ctx.memory is None or not getattr(ctx.memory, "enabled", False):
        return "Long-term memory is disabled."
    hits = ctx.memory.recall(query, k=limit)
    if not hits:
        return f"No memories matched {query!r}."
    return "\n".join(f"- {m.render()} (id {m.id}, score {m.score})" for m in hits)


@tool(
    "forget",
    "Delete a memory by id. Use this when the user says something you stored is wrong "
    "or out of date. Recall first to get the id.",
    obj(memory_id=s("The id returned by remember or recall.", required=True)),
)
def forget(ctx: ToolContext, memory_id: str) -> str:
    if ctx.memory is None or not getattr(ctx.memory, "enabled", False):
        return "Long-term memory is disabled."
    ctx.memory.forget(memory_id)
    return f"Deleted memory {memory_id}."
