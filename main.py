#!/usr/bin/env python3
"""Atlas - a personal Claude agent in your terminal.

    python main.py                  start a new session
    python main.py --resume LAST    continue the most recent session
    python main.py -p "question"    one-shot, print the answer, exit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from src.agent.loop import Agent  # noqa: E402
from src.agent.session import Session  # noqa: E402
from src.models.config import CONFIG  # noqa: E402
from src.utils.render import ConsoleRenderer  # noqa: E402

console = Console()

BANNER = r"""
  atlas
"""

HELP = """\
  /help              this message
  /memory [query]    list memories, or search them
  /forget <id>       delete a memory
  /sessions          list saved sessions
  /thinking          toggle showing the model's reasoning
  /cost              token and cost summary for this session
  /clear             start a fresh conversation (memory is kept)
  /exit              quit
"""


def _preflight() -> bool:
    """Check we have credentials before opening a REPL that cannot work."""
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    if (Path.home() / ".config" / "anthropic" / "credentials").exists():
        return True
    if list((Path.home() / ".config" / "anthropic").glob("credentials/*.json")):
        return True
    console.print(
        "[red]No Anthropic credentials found.[/]\n"
        "Set ANTHROPIC_API_KEY in .env (copy .env.example), or run [bold]ant auth login[/]."
    )
    return False


def _cmd_memory(agent: Agent, arg: str) -> None:
    if not agent.memory.enabled:
        reason = f" ({agent.memory.error})" if agent.memory.error else ""
        console.print(f"  [yellow]Long-term memory is disabled{reason}.[/]")
        return
    hits = agent.memory.recall(arg, k=20) if arg else agent.memory.all(limit=30)
    if not hits:
        console.print("  [dim]No memories stored yet.[/]")
        return
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("id", style="dim")
    table.add_column("when", style="dim")
    table.add_column("memory")
    for m in hits:
        table.add_row(m.id, m.created_at[:10], m.text[:100])
    console.print(table)
    console.print(f"  [dim]{agent.memory.count()} memories stored[/]")


def _cmd_sessions() -> None:
    rows = Session.list_recent()
    if not rows:
        console.print("  [dim]No saved sessions.[/]")
        return
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("id", style="dim")
    table.add_column("updated", style="dim")
    table.add_column("turns", justify="right", style="dim")
    table.add_column("title")
    for r in rows:
        table.add_row(r["id"], r["updated_at"][:16].replace("T", " "), str(r["turns"]), r["title"])
    console.print(table)


def _cmd_cost(agent: Agent) -> None:
    s = agent.tracer.summary()
    console.print(
        f"  [dim]turns[/] {s['turns']}   "
        f"[dim]in[/] {s['input_tokens']:,}   "
        f"[dim]out[/] {s['output_tokens']:,}   "
        f"[dim]cached[/] {s['cached_tokens']:,}   "
        f"[dim]cost[/] ${s['cost_usd']}"
    )
    console.print(f"  [dim]trace: {s['trace_file']}[/]")


def handle_command(line: str, agent: Agent, renderer: ConsoleRenderer) -> bool:
    """Returns False when the REPL should exit."""
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()

    if cmd in {"exit", "quit", "q"}:
        return False
    if cmd == "help":
        console.print(HELP)
    elif cmd == "memory":
        _cmd_memory(agent, arg)
    elif cmd == "forget":
        if not arg:
            console.print("  [yellow]Usage: /forget <memory-id>[/]")
        else:
            agent.memory.forget(arg)
            console.print(f"  [dim]Deleted {arg}.[/]")
    elif cmd == "sessions":
        _cmd_sessions()
    elif cmd == "thinking":
        renderer.show_thinking = not renderer.show_thinking
        console.print(f"  [dim]Reasoning display {'on' if renderer.show_thinking else 'off'}.[/]")
    elif cmd == "cost":
        _cmd_cost(agent)
    elif cmd == "clear":
        agent.session = Session()
        console.print("  [dim]New conversation. Long-term memory is unchanged.[/]")
    else:
        console.print(f"  [yellow]Unknown command /{cmd}. Try /help.[/]")
    return True


def repl(agent: Agent, renderer: ConsoleRenderer) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    from src.models.config import DATA_DIR

    prompt = PromptSession(history=FileHistory(str(DATA_DIR / ".input_history")))

    console.print(BANNER, style="bold")
    console.print(
        f"  [dim]{CONFIG.model} · effort {CONFIG.effort} · "
        f"{agent.memory.count()} memories · /help[/]\n"
    )

    while True:
        try:
            line = prompt.prompt("› ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, agent, renderer):
                break
            continue
        try:
            agent.run_turn(line)
        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted.[/]")
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            agent.tracer.emit("error", error=f"{type(exc).__name__}: {exc}")
            console.print(f"\n  [red]{type(exc).__name__}[/] {exc}")

    _cmd_cost(agent)


def main() -> int:
    parser = argparse.ArgumentParser(description="Atlas - a personal Claude agent.")
    parser.add_argument("-p", "--prompt", help="Run one prompt non-interactively and exit.")
    parser.add_argument(
        "--resume",
        metavar="ID",
        help="Continue a saved session. Use LAST for the most recent one.",
    )
    parser.add_argument("--thinking", action="store_true", help="Show the model's reasoning.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for the agent.")
    args = parser.parse_args()

    if not _preflight():
        return 1

    session = None
    if args.resume:
        target = args.resume
        if target.upper() == "LAST":
            recent = Session.list_recent(1)
            if not recent:
                console.print("[yellow]No saved sessions to resume.[/]")
                return 1
            target = recent[0]["id"]
        try:
            session = Session.load(target)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        console.print(f"[dim]Resumed session {session.id}: {session.title}[/]")

    renderer = ConsoleRenderer(show_thinking=args.thinking)
    agent = Agent(
        renderer=renderer,
        approve=renderer.approve,
        session=session,
        cwd=args.cwd,
    )

    if agent.memory.enabled:
        # Surfaces the one-time embedding-model download here, with a message,
        # instead of mid-conversation.
        with console.status("[dim]preparing local memory model...[/]", spinner="dots"):
            warm_error = agent.memory.warm()
        if warm_error:
            console.print(f"[yellow]Memory unavailable:[/] {warm_error}")
    elif agent.memory.error:
        console.print(f"[yellow]Memory unavailable:[/] {agent.memory.error}")

    if args.prompt:
        agent.run_turn(args.prompt)
        return 0

    repl(agent, renderer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
