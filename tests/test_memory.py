"""Memory tests.

The relevance filter is tested against the score distribution actually measured
from the local embedding model, using synthetic Memory objects so the suite
never needs the model download.
"""

import pytest

from src.agent.memory import Memory, MemoryStore
from src.models.config import CONFIG


def mem(score, label="m"):
    return Memory(id=label, text=label, tags=[], created_at="2026-01-01", score=score)


class TestRelevanceFilter:
    """Scores below are real values measured from all-MiniLM-L6-v2.

    The hard case: a correct hit can score 0.231 while an incorrect hit for a
    different query scores 0.363, so no single absolute cutoff works.
    """

    def test_keeps_only_the_strong_hit_when_one_dominates(self):
        # "where does sam live" -> correct 0.737, noise at 0.363 and 0.288
        kept = MemoryStore._filter(
            [mem(0.737, "right"), mem(0.363, "noise1"), mem(0.288, "noise2")],
            CONFIG.memory_min_score,
        )
        assert [m.id for m in kept] == ["right"]

    def test_keeps_a_weak_but_uncontested_hit(self):
        # "what minecraft work is in flight" -> correct 0.231, noise 0.079.
        # An absolute cutoff high enough to kill 0.363 noise would lose this.
        kept = MemoryStore._filter(
            [mem(0.231, "right"), mem(0.079, "noise")], CONFIG.memory_min_score
        )
        assert [m.id for m in kept] == ["right"]

    def test_returns_nothing_when_no_memory_is_relevant(self):
        # "what is the airspeed of a swallow" -> best hit 0.015
        assert MemoryStore._filter([mem(0.015)], CONFIG.memory_min_score) == []

    def test_handles_an_empty_result(self):
        assert MemoryStore._filter([], CONFIG.memory_min_score) == []

    def test_keeps_several_genuinely_close_hits(self):
        kept = MemoryStore._filter(
            [mem(0.60, "a"), mem(0.55, "b"), mem(0.12, "c")], CONFIG.memory_min_score
        )
        assert [m.id for m in kept] == ["a", "b"]


class TestDisabledMemory:
    def test_recall_is_a_noop_when_disabled(self):
        store = MemoryStore(enabled=False)
        assert store.recall("anything") == []
        assert store.count() == 0
        assert store.all() == []

    def test_remember_raises_when_disabled(self):
        store = MemoryStore(enabled=False)
        with pytest.raises(RuntimeError, match="disabled"):
            store.remember("x")

    def test_warm_is_a_noop_when_disabled(self):
        assert MemoryStore(enabled=False).warm() is None


class TestMemoryRendering:
    def test_render_includes_date_and_tags(self):
        out = Memory("i", "Sam likes tabs", ["prefs"], "2026-08-18T00:00:00").render()
        assert "2026-08-18" in out and "prefs" in out and "Sam likes tabs" in out
