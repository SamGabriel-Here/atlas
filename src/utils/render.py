"""Terminal rendering.

Model text is written to stdout raw rather than through rich's markup parser,
so a stray bracket in the output cannot be interpreted as a style tag.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.text import Text

console = Console()

TOOL_ICON = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "list_dir": "ls",
    "find_files": "find",
    "search_text": "grep",
    "run_command": "shell",
    "remember": "memory+",
    "recall": "memory?",
    "forget": "memory-",
    "workspace_info": "info",
    "web_search": "search",
    "web_fetch": "fetch",
}


class ConsoleRenderer:
    """Streams a turn to the terminal."""

    def __init__(self, show_thinking: bool = False) -> None:
        self.show_thinking = show_thinking
        self._mode: str | None = None
        self._wrote_any = False

    # --- streaming ----------------------------------------------------------

    def _switch(self, mode: str) -> None:
        if self._mode == mode:
            return
        if self._mode is not None:
            sys.stdout.write("\n")
        self._mode = mode
        if mode == "thinking":
            console.print("[dim italic]thinking[/]", highlight=False)
        elif mode == "text" and self._wrote_any:
            sys.stdout.write("\n")

    def on_block_start(self, kind: str, name: str | None) -> None:
        if kind == "server_tool_use" and name:
            self._end_line()
            console.print(f"  [cyan]{TOOL_ICON.get(name, name)}[/] [dim]running[/]", highlight=False)
        elif kind == "tool_use" and name:
            self._end_line()

    def on_thinking_delta(self, text: str) -> None:
        if not self.show_thinking:
            return
        self._switch("thinking")
        sys.stdout.write(Text(text).plain)
        sys.stdout.flush()
        self._wrote_any = True

    def on_text_delta(self, text: str) -> None:
        self._switch("text")
        sys.stdout.write(text)
        sys.stdout.flush()
        self._wrote_any = True

    # --- structured events --------------------------------------------------

    def _end_line(self) -> None:
        if self._mode is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._mode = None

    def on_tool_result(self, name: str, result: str, ok: bool) -> None:
        self._end_line()
        label = TOOL_ICON.get(name, name)
        first = (result or "").strip().splitlines()
        summary = first[0][:110] if first else ""
        colour = "cyan" if ok else "red"
        console.print(f"  [{colour}]{label}[/] [dim]{summary}[/]", highlight=False)

    def on_notice(self, message: str) -> None:
        self._end_line()
        console.print(f"  [yellow]note[/] {message}", highlight=False)

    def on_turn_end(self, cost: float, usage: Any) -> None:
        self._end_line()
        console.print(f"[dim]  ${cost:.4f} this session[/]", highlight=False)
        self._wrote_any = False

    # --- prompts ------------------------------------------------------------

    def approve(self, command: str, reason: str) -> bool:
        """Ask the user before running an unrecognized shell command."""
        self._end_line()
        console.print()
        console.print("  [bold yellow]shell approval[/]", highlight=False)
        console.print(f"  [white]{command}[/]", highlight=False)
        console.print(f"  [dim]{reason}[/]", highlight=False)
        try:
            answer = console.input("  run it? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        return answer in {"y", "yes"}
