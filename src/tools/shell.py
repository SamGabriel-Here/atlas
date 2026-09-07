"""Shell execution, gated by the approval policy in guardrails."""

from __future__ import annotations

import subprocess

from src.models.config import CONFIG
from src.tools.registry import ToolContext, i, obj, s, tool
from src.utils.guardrails import GuardrailError, classify_command, safe_path

MAX_OUTPUT = 30_000
DEFAULT_TIMEOUT = 120


@tool(
    "run_command",
    "Run a shell command and return its combined output. Use this for things the other "
    "tools cannot do: running tests, git operations, build steps, invoking CLI tools. "
    "Destructive commands are blocked and unrecognized ones require the user's approval, "
    "so prefer the narrower file tools when they would do the job.",
    obj(
        command=s("The command to run, as you would type it in a shell.", required=True),
        cwd=s("Directory to run in. Must be inside the allowed workspace."),
        timeout=i(f"Seconds before the command is killed. Defaults to {DEFAULT_TIMEOUT}."),
    ),
)
def run_command(
    ctx: ToolContext,
    command: str,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    verdict, reason = classify_command(command)

    if verdict == "deny":
        raise GuardrailError(f"Refused: {reason}")

    if verdict == "ask":
        if ctx.approve is None:
            raise GuardrailError(
                f"Refused: {reason} No approval channel is available in this context."
            )
        if not ctx.approve(command=command, reason=reason):
            return "The user declined to run this command. Do not retry it; ask what they would prefer."

    workdir = safe_path(cwd, must_exist=True) if cwd else safe_path(ctx.cwd, must_exist=True)
    timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), 600))

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s and was killed: {command}"

    output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + f"\n...<clipped {len(output) - MAX_OUTPUT} chars>"
    status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
    return f"$ {command}  ({status}, in {workdir})\n{output or '(no output)'}"


@tool(
    "workspace_info",
    "Report which directories the agent is allowed to touch and how shell commands are "
    "gated. Call this when a path is refused, or when you are unsure what you can reach.",
    obj(),
)
def workspace_info(ctx: ToolContext) -> str:
    roots = "\n".join(f"  - {r}" for r in CONFIG.workspace_roots)
    allow = ", ".join(sorted(CONFIG.shell_allowlist)) or "(none)"
    return (
        f"Allowed workspace roots:\n{roots}\n"
        f"Shell mode: {CONFIG.shell_mode}\n"
        f"Auto-approved commands: {allow}\n"
        f"Current working directory: {ctx.cwd}"
    )
