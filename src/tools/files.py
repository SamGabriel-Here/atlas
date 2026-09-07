"""Filesystem tools. Every path is confined by guardrails.safe_path."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from src.tools.registry import ToolContext, arr, b, i, obj, s, tool
from src.utils.guardrails import GuardrailError, safe_path

MAX_READ_BYTES = 400_000
MAX_MATCHES = 200
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(4096)
    except OSError:
        return False


@tool(
    "read_file",
    "Read a text file from disk. Call this whenever you need the actual contents of a "
    "file rather than guessing at them - before editing, reviewing, or answering a "
    "question about a specific file. Returns numbered lines.",
    obj(
        path=s("Absolute or ~-relative path to the file.", required=True),
        start_line=i("1-indexed first line to return. Omit to start at the top."),
        end_line=i("1-indexed last line to return. Omit to read to the end."),
    ),
)
def read_file(ctx: ToolContext, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    target = safe_path(path, must_exist=True)
    if target.is_dir():
        raise GuardrailError(f"{target} is a directory. Use list_dir instead.")
    if _is_probably_binary(target):
        return f"{target} looks like a binary file ({target.stat().st_size} bytes); not reading it as text."
    if target.stat().st_size > MAX_READ_BYTES:
        return (
            f"{target} is {target.stat().st_size} bytes, over the {MAX_READ_BYTES} byte limit. "
            "Re-read it with start_line/end_line to page through it."
        )

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    lo = max(1, start_line or 1)
    hi = min(len(lines), end_line or len(lines))
    if lo > len(lines):
        return f"{target} has {len(lines)} lines; start_line={lo} is past the end."
    body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(lo, hi + 1))
    return f"{target} (lines {lo}-{hi} of {len(lines)})\n{body}"


@tool(
    "write_file",
    "Create a new file or completely replace an existing one. For a targeted change to "
    "part of an existing file, prefer edit_file - this overwrites everything. Creates "
    "parent directories as needed.",
    obj(
        path=s("Where to write.", required=True),
        content=s("The full file contents.", required=True),
    ),
)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = safe_path(path)
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {target} ({len(content)} chars, {content.count(chr(10)) + 1} lines)."


@tool(
    "edit_file",
    "Replace one exact string in a file with another. Use this for surgical changes to "
    "existing files. old_string must appear exactly once - if it appears zero times or "
    "more than once the edit is refused, so include enough surrounding context to be unique.",
    obj(
        path=s("File to edit.", required=True),
        old_string=s("Exact text to replace, including whitespace and indentation.", required=True),
        new_string=s("Replacement text.", required=True),
    ),
)
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str) -> str:
    target = safe_path(path, must_exist=True)
    original = target.read_text(encoding="utf-8")
    hits = original.count(old_string)
    if hits == 0:
        return f"No match in {target}. The old_string was not found - re-read the file and copy the exact text."
    if hits > 1:
        return f"old_string appears {hits} times in {target}. Add surrounding context to make it unique."
    target.write_text(original.replace(old_string, new_string, 1), encoding="utf-8")
    return f"Edited {target}: replaced 1 occurrence."


@tool(
    "list_dir",
    "List the entries of a directory. Use this to orient yourself in an unfamiliar "
    "part of the filesystem before reading or searching.",
    obj(
        path=s("Directory to list.", required=True),
        show_hidden=b("Include dotfiles. Defaults to false."),
    ),
)
def list_dir(ctx: ToolContext, path: str, show_hidden: bool = False) -> str:
    target = safe_path(path, must_exist=True)
    if not target.is_dir():
        raise GuardrailError(f"{target} is not a directory.")
    rows = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if not show_hidden and entry.name.startswith("."):
            continue
        try:
            size = "" if entry.is_dir() else f"{entry.stat().st_size:>10}"
        except OSError:
            size = ""
        rows.append(f"{'dir ' if entry.is_dir() else 'file'} {size:>10}  {entry.name}")
    return f"{target} ({len(rows)} entries)\n" + ("\n".join(rows) if rows else "(empty)")


@tool(
    "find_files",
    "Find files by glob pattern under a directory, recursively. Use this when you know "
    "roughly what a file is called but not where it lives.",
    obj(
        pattern=s("Glob pattern matched against the filename, e.g. '*.py' or 'test_*.md'.", required=True),
        path=s("Directory to search under.", required=True),
        limit=i("Max results. Defaults to 100."),
    ),
)
def find_files(ctx: ToolContext, pattern: str, path: str, limit: int = 100) -> str:
    root = safe_path(path, must_exist=True)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                found.append(str(Path(dirpath) / name))
                if len(found) >= limit:
                    return f"{len(found)} matches (limit reached):\n" + "\n".join(found)
    return f"{len(found)} matches:\n" + ("\n".join(found) if found else "(none)")


@tool(
    "search_text",
    "Search file contents for a regular expression across a directory tree. Use this to "
    "locate where something is defined or referenced when you do not know the filename.",
    obj(
        pattern=s("Python regular expression.", required=True),
        path=s("Directory to search under.", required=True),
        file_glob=s("Only search files matching this glob, e.g. '*.py'. Defaults to all text files."),
        ignore_case=b("Case-insensitive match. Defaults to false."),
    ),
)
def search_text(
    ctx: ToolContext,
    pattern: str,
    path: str,
    file_glob: str = "*",
    ignore_case: bool = False,
) -> str:
    root = safe_path(path, must_exist=True)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"Invalid regular expression: {exc}"

    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if not fnmatch.fnmatch(name, file_glob):
                continue
            fpath = Path(dirpath) / name
            if _is_probably_binary(fpath):
                continue
            try:
                for lineno, line in enumerate(
                    fpath.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        matches.append(f"{fpath}:{lineno}: {line.strip()[:200]}")
                        if len(matches) >= MAX_MATCHES:
                            return f"{len(matches)} matches (limit reached):\n" + "\n".join(matches)
            except OSError:
                continue
    return f"{len(matches)} matches:\n" + ("\n".join(matches) if matches else "(none)")
