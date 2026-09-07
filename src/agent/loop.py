"""The agent loop.

A manual loop rather than the SDK tool runner, for one specific reason: this
agent mixes client-side tools with server-side web search, and a server-tool
turn can end with stop_reason == "pause_turn". The Python tool runner exits
silently on a paused turn and returns it as the final message, which looks
like a complete answer but is truncated. Here it is handled explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import anthropic

from src.agent.memory import MemoryStore
from src.agent.session import Session
from src.models.config import CONFIG
from src.prompts.system import build_system_prompt, memory_preamble
from src.tools import files, memory_tools, shell  # noqa: F401 - registers the tools
from src.tools.registry import ToolContext, client_tool_definitions, dispatch
from src.tools.server import WEB_TOOLS
from src.utils.logging import Timer, Tracer

MAX_TOOL_ITERATIONS = 40
MAX_PAUSE_RESUMES = 5


class Renderer(Protocol):
    def on_thinking_delta(self, text: str) -> None: ...
    def on_text_delta(self, text: str) -> None: ...
    def on_block_start(self, kind: str, name: str | None) -> None: ...
    def on_tool_result(self, name: str, result: str, ok: bool) -> None: ...
    def on_notice(self, message: str) -> None: ...
    def on_turn_end(self, cost: float, usage: Any) -> None: ...


class Agent:
    def __init__(
        self,
        renderer: Renderer,
        approve: Callable[..., bool],
        session: Session | None = None,
        cwd: str = ".",
        user: str | None = None,
    ) -> None:
        self.client = anthropic.Anthropic()
        self.renderer = renderer
        self.session = session or Session()
        self.memory = MemoryStore()
        self.tracer = Tracer(self.session.id)
        self.cwd = cwd
        self.system = build_system_prompt(cwd=cwd, user=user)
        self.ctx = ToolContext(memory=self.memory, approve=approve, cwd=cwd)
        self.tools = client_tool_definitions() + WEB_TOOLS
        # Set to False the first time the API rejects a role=system message, so
        # a model that does not support them falls back gracefully.
        self.supports_system_messages = True

    def _params(self) -> dict[str, Any]:
        return {
            "model": CONFIG.model,
            "max_tokens": CONFIG.max_tokens,
            # A breakpoint on the last system block caches tools + system, which
            # are byte-identical across every turn and every session.
            "system": [
                {
                    "type": "text",
                    "text": self.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            # Auto-placed on the last cacheable block, so the conversation
            # prefix is reused turn over turn.
            "cache_control": {"type": "ephemeral"},
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": CONFIG.effort},
            "tools": self.tools,
            "messages": self.session.messages,
        }

    def _stream_once(self):
        with self.client.messages.stream(**self._params()) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    self.renderer.on_block_start(block.type, getattr(block, "name", None))
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.renderer.on_thinking_delta(delta.thinking)
                    elif delta.type == "text_delta":
                        self.renderer.on_text_delta(delta.text)
            return stream.get_final_message()

    def _request(self):
        """Send one request, retrying without role=system messages if rejected."""
        try:
            return self._stream_once()
        except anthropic.BadRequestError as exc:
            if self.supports_system_messages and "system" in str(exc).lower():
                self.supports_system_messages = False
                self.session.drop_system_messages()
                self.renderer.on_notice(
                    f"{CONFIG.model} does not accept mid-conversation system messages; "
                    "folding memory into the user turn instead."
                )
                self.tracer.emit("system_message_unsupported", error=str(exc))
                return self._stream_once()
            raise

    def _inject_memories(self, user_input: str) -> None:
        if not self.memory.enabled:
            return
        hits = self.memory.recall(user_input)
        if not hits:
            return
        self.tracer.emit("memory_recall", query=user_input, count=len(hits),
                         ids=[m.id for m in hits])
        preamble = memory_preamble(hits)
        if self.supports_system_messages:
            self.session.add_system(preamble)
        else:
            # Fold into the user turn: replace the message we just appended.
            last = self.session.messages[-1]
            last["content"] = f"{preamble}\n\n---\n\n{last['content']}"

    def run_turn(self, user_input: str) -> str:
        """Run one user turn to completion. Returns the final assistant text."""
        self.tracer.turn += 1
        self.tracer.emit("user_message", text=user_input)
        self.session.add_user(user_input)
        self._inject_memories(user_input)

        final_text: list[str] = []
        pauses = 0

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._request()
            self.tracer.record_usage(response.usage)
            self.session.add_assistant(response.content)

            for block in response.content:
                if block.type == "text":
                    final_text.append(block.text)

            stop = response.stop_reason

            if stop == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details else None
                self.tracer.emit("refusal", category=category)
                msg = (
                    "The model declined this request"
                    + (f" ({category})" if category else "")
                    + ". Nothing was produced. Try rephrasing, or narrow the ask."
                )
                self.renderer.on_notice(msg)
                self.session.save()
                return msg

            if stop == "pause_turn":
                pauses += 1
                if pauses > MAX_PAUSE_RESUMES:
                    self.renderer.on_notice(
                        "The turn kept pausing on server-side tools and was stopped after "
                        f"{MAX_PAUSE_RESUMES} resumes. The answer may be incomplete."
                    )
                    break
                # Re-send with the paused assistant turn appended; the server
                # picks up where it left off. No extra user message.
                self.tracer.emit("pause_turn_resume", attempt=pauses)
                continue

            if stop == "max_tokens":
                self.renderer.on_notice(
                    f"Hit the {CONFIG.max_tokens} token ceiling mid-response; the answer is "
                    "cut off. Raise ATLAS_MAX_TOKENS or ask for a narrower slice."
                )
                break

            if stop != "tool_use":
                break

            results = self._run_tools(response)
            if not results:
                # stop_reason said tool_use but nothing client-side to run.
                break
            self.session.add_tool_results(results)

        else:
            self.renderer.on_notice(
                f"Stopped after {MAX_TOOL_ITERATIONS} tool rounds without finishing."
            )

        self.session.save()
        self.renderer.on_turn_end(self.tracer.total_cost, None)
        return "\n".join(final_text).strip()

    def _run_tools(self, response) -> list[dict[str, Any]]:
        """Execute every client-side tool_use block in one assistant turn.

        Server-side tool blocks (web search, web fetch) are already resolved by
        the API and must not get a tool_result.
        """
        results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            with Timer() as timer:
                text, ok = dispatch(block.name, dict(block.input or {}), self.ctx)
            self.tracer.tool_call(block.name, block.input, text, ok, timer.elapsed)
            self.renderer.on_tool_result(block.name, text, ok)
            entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": text,
            }
            if not ok:
                entry["is_error"] = True
            results.append(entry)
        return results
