"""Conversation state and its on-disk form.

The API is stateless, so the full message list is the session. Assistant
content blocks are stored verbatim - including thinking blocks, which must be
passed back unchanged for the model to continue a turn correctly.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.models.config import SESSION_DIR


def _to_jsonable(value: Any) -> Any:
    """Convert SDK content blocks to plain data without losing fields."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    title: str = ""

    @property
    def path(self):
        return SESSION_DIR / f"{self.id}.json"

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        if not self.title:
            self.title = text[:70]

    def add_system(self, text: str) -> None:
        """Mid-conversation operator context.

        Must follow a user message and be either last or followed by an
        assistant turn - which is exactly how it is used here.
        """
        self.messages.append({"role": "system", "content": text})

    def add_assistant(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": _to_jsonable(content)})

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": results})

    def drop_system_messages(self) -> None:
        """Remove role=system entries, for models that do not accept them."""
        self.messages = [m for m in self.messages if m.get("role") != "system"]

    def save(self) -> None:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": _to_jsonable(self.messages),
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, session_id: str) -> "Session":
        path = SESSION_DIR / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No saved session {session_id!r} in {SESSION_DIR}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            id=data["id"],
            messages=data.get("messages", []),
            created_at=data.get("created_at", ""),
            title=data.get("title", ""),
        )

    @staticmethod
    def list_recent(limit: int = 20) -> list[dict[str, Any]]:
        if not SESSION_DIR.exists():
            return []
        rows = []
        for path in SESSION_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "id": data.get("id", path.stem),
                    "title": data.get("title", ""),
                    "updated_at": data.get("updated_at", ""),
                    "turns": sum(1 for m in data.get("messages", []) if m.get("role") == "user"),
                }
            )
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)[:limit]
