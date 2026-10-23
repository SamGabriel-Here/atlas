"""Guardrails: path confinement, shell gating, and secret redaction.

Tool inputs are model output, not trusted input. Every path and command
crosses this module before it reaches the filesystem or a shell.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from src.models.config import CONFIG, PROJECT_ROOT


class GuardrailError(Exception):
    """Raised when a tool call is refused. Surfaced to the model as a tool error."""


# Commands that are never auto-approved, even if the first word is allowlisted.
# Anchored to the start of a command: line start, a shell operator, a newline,
# a subshell opener, or an xargs prefix, so `ls\nrm -rf ~` and `$(rm -rf ~)`
# are caught too.
_DESTRUCTIVE = re.compile(
    r"""(?x)
    (^|[;&|\n(`]|\bxargs\s+(-\S+\s+)*)\s*
    (rm\b[^;&|\n]*\s(-[a-zA-Z]*[rRf][a-zA-Z]*|--recursive|--force)\b  # rm -r, -f, -rf, -R
    |sudo\b
    |mkfs\b
    |dd\s+.*\bof=
    |shutdown\b|reboot\b|halt\b
    |chmod\s+(-R\s+)?777\b
    |curl\b[^|]*\|\s*(ba)?sh   # curl ... | sh
    |wget\b[^|]*\|\s*(ba)?sh
    |:\(\)\s*\{.*\};:          # fork bomb
    |git\s+push\b.*(--force|\s-f\b)
    |git\s+reset\b.*--hard
    |git\s+clean\b.*\s-[a-zA-Z]*f
    |find\b.*\s-delete\b
    |>\s*/dev/(sd|disk|nvme)
    )""",
)

# Shell metacharacters that can chain, redirect or substitute a second command.
_SHELL_OPERATORS = re.compile(r"[;&|><`$\n\r]")

# Programs that run arbitrary code or install packages. Never auto-approved,
# even if a user's ATLAS_SHELL_ALLOWLIST still names them.
_NEVER_AUTO = {
    "python", "python3", "pip", "pip3", "pytest", "node", "npm", "npx",
    "bash", "sh", "zsh", "perl", "ruby", "env", "xargs", "eval", "exec",
}

# git subcommands that only read. Anything else (checkout, commit, stash, ...)
# can discard or rewrite work, so it asks.
_GIT_READONLY = {
    "status", "log", "diff", "show", "blame", "ls-files", "rev-parse",
    "shortlog", "describe", "grep",
}

# Flags that make an otherwise read-only command run another program.
_EXEC_FLAGS = {
    "find": {"-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"},
    "rg": {"--pre"},
}

# Patterns that look like credentials. Redacted before anything is written to
# the trace log, so a leaked key in tool output does not become a leaked key on disk.
_SECRETS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)\b(api[_\-]?key|secret|password|token)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})"),
]


def redact(text: str) -> str:
    """Mask anything that looks like a credential."""
    if not text:
        return text
    for pat in _SECRETS:
        if pat.groups >= 2:
            text = pat.sub(lambda m: f"{m.group(1)}=<redacted>", text)
        else:
            text = pat.sub("<redacted>", text)
    return text


def _inside_roots(resolved: Path) -> bool:
    return any(resolved == root or root in resolved.parents for root in CONFIG.workspace_roots)


def safe_path(raw: str, *, must_exist: bool = False, write: bool = False) -> Path:
    """Resolve a model-supplied path and confine it to the configured roots.

    Resolution happens before the containment check, so `..` segments and
    symlinks that point outside a root are caught rather than followed.

    With write=True, two more places are refused even inside a root: `.git`
    directories (hooks and config there run code on the next git command) and
    Atlas's own directory (its .env and source decide what the guardrails allow).
    """
    if not raw or not raw.strip():
        raise GuardrailError("Empty path.")

    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise GuardrailError(f"Could not resolve path {raw!r}: {exc}") from exc

    if not _inside_roots(resolved):
        allowed = ", ".join(str(r) for r in CONFIG.workspace_roots)
        raise GuardrailError(
            f"Path {resolved} is outside the allowed workspace. Allowed roots: {allowed}"
        )

    if write:
        if ".git" in resolved.parts:
            raise GuardrailError(f"Refusing to write inside a .git directory: {resolved}")
        if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
            raise GuardrailError(
                f"Refusing to write to Atlas's own files ({PROJECT_ROOT}); "
                "they control its guardrails."
            )

    if must_exist and not resolved.exists():
        raise GuardrailError(f"No such file or directory: {resolved}")
    return resolved


def _outside_path_arg(args: list[str], cwd: str | None) -> str | None:
    """Return the first argument that names a path outside the workspace, if any."""
    base = Path(cwd).expanduser() if cwd else Path.cwd()
    for arg in args:
        values = [arg]
        if arg.startswith("-") and "=" in arg:
            values.append(arg.split("=", 1)[1])  # --file=/etc/hosts
        for value in values:
            if not value.startswith(("/", "~", ".")) and "/" not in value:
                continue  # a bare name or flag, resolved inside cwd
            try:
                resolved = (base / Path(value).expanduser()).resolve()
            except (OSError, RuntimeError):
                return value
            if not _inside_roots(resolved):
                return value
    return None


def classify_command(command: str, cwd: str | None = None) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is allow | ask | deny.

    cwd is where the command will run; relative path arguments resolve from it.
    """
    stripped = command.strip()
    if not stripped:
        return "deny", "Empty command."

    if _DESTRUCTIVE.search(stripped):
        return "deny", "Command matches a destructive pattern and is blocked."

    if CONFIG.shell_mode == "yolo":
        return "allow", "shell_mode=yolo"

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return "ask", "Command could not be parsed; asking for confirmation."

    first = Path(tokens[0]).name
    ask_reason = None

    if first not in CONFIG.shell_allowlist:
        ask_reason = f"{first} is not allowlisted."
    elif first in _NEVER_AUTO:
        ask_reason = f"{first} runs arbitrary code, so it always asks."
    elif _SHELL_OPERATORS.search(stripped):
        ask_reason = "The command contains shell operators."
    elif first == "git" and (len(tokens) < 2 or tokens[1] not in _GIT_READONLY):
        # Also catches `git -c core.pager=...`, where an option comes before the subcommand.
        ask_reason = "Only read-only git subcommands run without asking."
    elif _EXEC_FLAGS.get(first, set()) & {t.split("=", 1)[0] for t in tokens[1:]}:
        ask_reason = f"This {first} invocation runs another program."
    elif outside := _outside_path_arg(tokens[1:], cwd):
        ask_reason = f"{outside} is outside the allowed workspace."

    if ask_reason is None:
        return "allow", f"{first} is allowlisted and the command stays inside the workspace."

    if CONFIG.shell_mode == "allowlist":
        return "deny", f"{ask_reason} shell_mode=allowlist denies anything that would ask."

    return "ask", ask_reason
