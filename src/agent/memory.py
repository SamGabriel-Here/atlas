"""Long-term memory: a local vector store that survives across sessions.

Chroma runs embedded with a bundled local embedding model, so memory works
offline and needs no second API key. Nothing here leaves the machine.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.models.config import CONFIG, MEMORY_DIR


@dataclass
class Memory:
    id: str
    text: str
    tags: list[str]
    created_at: str
    score: float | None = None

    def render(self) -> str:
        tags = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"({self.created_at[:10]}){tags} {self.text}"


class MemoryStore:
    """Persistent vector memory. Degrades to a no-op if Chroma is unavailable."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = CONFIG.memory_enabled if enabled is None else enabled
        self._collection = None
        self.error: str | None = None
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(MEMORY_DIR),
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
            self._collection = client.get_or_create_collection(
                name="memories", metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:  # noqa: BLE001 - memory is optional, never fatal
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"

    # --- writes -------------------------------------------------------------

    def warm(self) -> str | None:
        """Force the local embedding model to load.

        Chroma downloads its embedding model (~80MB) on first use. Doing that
        lazily means the download lands in the middle of a conversation, with a
        progress bar in the transcript or a timeout mid-turn. Calling this at
        startup moves it somewhere the user can see it. Returns an error string
        if the model could not be prepared.
        """
        if not self.enabled or self._collection is None:
            return None
        try:
            self._collection.query(query_texts=["warmup"], n_results=1)
            return None
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"
            return self.error

    def remember(self, text: str, tags: list[str] | None = None) -> str:
        if not self.enabled or self._collection is None:
            raise RuntimeError("Long-term memory is disabled.")
        text = text.strip()
        if not text:
            raise ValueError("Refusing to store an empty memory.")

        mem_id = uuid.uuid4().hex[:16]
        try:
            self._collection.add(
                ids=[mem_id],
                documents=[text],
                metadatas=[
                    {
                        "tags": ",".join(tags or []),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - do not surface a raw traceback
            raise RuntimeError(
                f"Could not write to long-term memory ({type(exc).__name__}). "
                "The local embedding model may not be available."
            ) from exc
        return mem_id

    def forget(self, mem_id: str) -> None:
        if not self.enabled or self._collection is None:
            raise RuntimeError("Long-term memory is disabled.")
        self._collection.delete(ids=[mem_id])

    # --- reads --------------------------------------------------------------

    def recall(self, query: str, k: int | None = None, min_score: float | None = None) -> list[Memory]:
        """Nearest memories above the relevance floor.

        The floor matters: without it every turn injects the top-k memories no
        matter how unrelated, which is just noise in the context window.
        """
        if not self.enabled or self._collection is None or not query.strip():
            return []
        k = k or CONFIG.memory_top_k
        floor = CONFIG.memory_min_score if min_score is None else min_score
        try:
            res = self._collection.query(query_texts=[query], n_results=k)
        except Exception:  # noqa: BLE001 - an empty collection or backend hiccup
            return []
        hits = self._to_memories(res)
        return self._filter(hits, floor)

    @staticmethod
    def _filter(hits: list[Memory], floor: float) -> list[Memory]:
        """Two-stage relevance filter.

        Embedding scores for short queries sit in a narrow band, and a correct
        hit for one query can score lower than an incorrect hit for another --
        so a single absolute cutoff either leaks noise or drops real matches.
        The absolute floor answers "is anything relevant here at all", and the
        relative floor then keeps only what is competitive with the best hit.
        """
        scored = [m for m in hits if m.score is not None]
        if not scored:
            return hits[:1] if hits else []
        best = max(m.score for m in scored)
        if best < floor:
            return []
        cutoff = best * CONFIG.memory_rel_ratio
        return [m for m in scored if m.score >= cutoff]

    def all(self, limit: int = 100) -> list[Memory]:
        if not self.enabled or self._collection is None:
            return []
        res = self._collection.get(limit=limit)
        out = []
        for mem_id, doc, meta in zip(
            res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
        ):
            out.append(self._build(mem_id, doc, meta, None))
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def count(self) -> int:
        if not self.enabled or self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:  # noqa: BLE001
            return 0

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _build(mem_id: str, doc: str, meta: dict[str, Any] | None, dist: float | None) -> Memory:
        meta = meta or {}
        raw_tags = meta.get("tags") or ""
        return Memory(
            id=mem_id,
            text=doc,
            tags=[t for t in raw_tags.split(",") if t],
            created_at=meta.get("created_at", ""),
            # Chroma returns cosine distance; flip it so higher means closer.
            score=None if dist is None else round(1.0 - dist, 4),
        )

    def _to_memories(self, res: dict[str, Any]) -> list[Memory]:
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for i, mem_id in enumerate(ids):
            out.append(
                self._build(
                    mem_id,
                    docs[i] if i < len(docs) else "",
                    metas[i] if i < len(metas) else {},
                    dists[i] if i < len(dists) else None,
                )
            )
        return out
