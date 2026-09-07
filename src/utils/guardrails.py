"""Guardrails: path confinement, shell gating, and secret redaction.

Tool inputs are model output, not trusted input. Every path and command
crosses this module before it reaches the filesystem or a shell.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from src.models.config import CONFIG


class GuardrailError(Exception):
    """Raised when a tool call is refused. Surfaced to the model as a tool error."""


# Commands that are never auto-approved, even if the first word is allowlisted.
_DESTRUCTIVE = re.compile(
    r"""(?x)
    (^|[;&|]\s*)\s*(rm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf]  # rm -rf and friends
    |sudo\b
    |mkfs\b
    |dd\s+.*\bof=
    |shutdown\b|reboot\b|halt\b
    |chmod\s+(-R\s+)?777\b
    |curl\b[^|]*\|\s*(ba)?sh   # curl ... | sh
    |wget\b[^|]*\|\s*(ba)?sh
    |:\(\)\s*\{.*\};:          # fork bomb
    |git\s+push\b.*--force
    |>\s*/dev/(sd|disk|nvme)
    )""",
)

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


def safe_path(raw: str, *, must_exist: bool = False) -> Path:
    """Resolve a model-supplied path and confine it to the configured roots.

    Resolution happens before the containment check, so `..` segments and
    symlinks that point outside a root are caught rather than followed.
    """
    if not raw or not raw.strip():
        raise GuardrailError("Empty path.")

    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise GuardrailError(f"Could not resolve path {raw!r}: {exc}") from exc

    for root in CONFIG.workspace_roots:
        if resolved == root or root in resolved.parents:
            break
    else:
        allowed = ", ".join(str(r) for r in CONFIG.workspace_roots)
        raise GuardrailError(
            f"Path {resolved} is outside the allowed workspace. Allowed roots: {allowed}"
        )

    if must_exist and not resolved.exists():
        raise GuardrailError(f"No such file or directory: {resolved}")
    return resolved


def classify_command(command: str) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is allow | ask | deny."""
    stripped = command.strip()
    if not stripped:
        return "deny", "Empty command."

    if _DESTRUCTIVE.search(stripped):
        return "deny", "Command matches a destructive pattern and is blocked."

    if CONFIG.shell_mode == "yolo":
        return "allow", "shell_mode=yolo"

    try:
        first = shlex.split(stripped)[0]
    except ValueError:
        return "ask", "Command could not be parsed; asking for confirmation."

    first = Path(first).name
    if first in CONFIG.shell_allowlist and not re.search(r"[;&|><`$]", stripped):
        return "allow", f"{first} is allowlisted and the command has no shell operators."

    if CONFIG.shell_mode == "allowlist":
        return "deny", f"{first} is not in ATLAS_SHELL_ALLOWLIST and shell_mode=allowlist."

    return "ask", f"{first} is not allowlisted."
