"""Tests for the loop's control flow, with a stubbed model.

These exercise the parts most likely to break silently: pause_turn resumption,
refusal handling, and not sending tool_results for server-side tools.
"""

from types import SimpleNamespace

import pytest

from src.agent.session import Session


def block(**kw):
    return SimpleNamespace(**kw)


def response(content, stop_reason, **extra):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        **extra,
    )


class RecordingRenderer:
    def __init__(self):
        self.notices = []
        self.tool_results = []

    def on_thinking_delta(self, text): pass
    def on_text_delta(self, text): pass
    def on_block_start(self, kind, name): pass
    def on_tool_result(self, name, result, ok): self.tool_results.append((name, ok))
    def on_notice(self, message): self.notices.append(message)
    def on_turn_end(self, cost, usage): pass


@pytest.fixture
def agent(workspace, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    from src.agent.loop import Agent

    renderer = RecordingRenderer()
    a = Agent(
        renderer=renderer,
        approve=lambda command, reason: False,
        session=Session(),
        cwd=str(workspace),
    )
    a.memory.enabled = False  # keep tests off the vector store
    a.renderer = renderer
    monkeypatch.setattr(a.session, "save", lambda: None)
    return a


class TestRunTurn:
    def test_plain_answer_is_returned(self, agent, monkeypatch):
        monkeypatch.setattr(
            agent, "_request", lambda: response([block(type="text", text="42")], "end_turn")
        )
        assert agent.run_turn("what is 6 times 7") == "42"

    def test_pause_turn_is_resumed_not_treated_as_final(self, agent, monkeypatch):
        """A paused server-tool turn must continue, not silently truncate."""
        calls = []

        def fake():
            calls.append(1)
            if len(calls) == 1:
                return response([block(type="text", text="searching")], "pause_turn")
            return response([block(type="text", text="done")], "end_turn")

        monkeypatch.setattr(agent, "_request", fake)
        out = agent.run_turn("look something up")
        assert len(calls) == 2, "the paused turn was not resumed"
        assert "done" in out

    def test_runaway_pausing_stops_and_warns(self, agent, monkeypatch):
        monkeypatch.setattr(
            agent, "_request", lambda: response([block(type="text", text="x")], "pause_turn")
        )
        agent.run_turn("loop forever")
        assert any("pausing" in n for n in agent.renderer.notices)

    def test_refusal_is_surfaced_without_reading_content(self, agent, monkeypatch):
        monkeypatch.setattr(
            agent,
            "_request",
            lambda: response([], "refusal", stop_details=SimpleNamespace(category="cyber")),
        )
        out = agent.run_turn("something disallowed")
        assert "declined" in out and "cyber" in out

    def test_max_tokens_warns_about_truncation(self, agent, monkeypatch):
        monkeypatch.setattr(
            agent, "_request", lambda: response([block(type="text", text="cut")], "max_tokens")
        )
        agent.run_turn("write forever")
        assert any("cut off" in n for n in agent.renderer.notices)

    def test_client_tool_result_is_fed_back(self, agent, monkeypatch, workspace):
        target = workspace / "x.txt"
        target.write_text("contents")
        calls = []

        def fake():
            calls.append(1)
            if len(calls) == 1:
                return response(
                    [block(type="tool_use", id="t1", name="read_file",
                           input={"path": str(target)})],
                    "tool_use",
                )
            return response([block(type="text", text="read it")], "end_turn")

        monkeypatch.setattr(agent, "_request", fake)
        agent.run_turn("read x.txt")

        results = [m for m in agent.session.messages if m["role"] == "user"
                   and isinstance(m["content"], list)]
        assert len(results) == 1
        assert results[0]["content"][0]["tool_use_id"] == "t1"
        assert "contents" in results[0]["content"][0]["content"]

    def test_failing_tool_is_marked_is_error(self, agent, monkeypatch):
        calls = []

        def fake():
            calls.append(1)
            if len(calls) == 1:
                return response(
                    [block(type="tool_use", id="t1", name="read_file",
                           input={"path": "/etc/passwd"})],
                    "tool_use",
                )
            return response([block(type="text", text="blocked")], "end_turn")

        monkeypatch.setattr(agent, "_request", fake)
        agent.run_turn("read /etc/passwd")
        results = [m for m in agent.session.messages if m["role"] == "user"
                   and isinstance(m["content"], list)]
        assert results[0]["content"][0]["is_error"] is True

    def test_server_tool_blocks_get_no_tool_result(self, agent, monkeypatch):
        """web_search resolves server-side; sending a tool_result for it is a 400."""
        monkeypatch.setattr(
            agent,
            "_request",
            lambda: response(
                [
                    block(type="server_tool_use", id="s1", name="web_search", input={"query": "x"}),
                    block(type="web_search_tool_result", tool_use_id="s1", content=[]),
                    block(type="text", text="found it"),
                ],
                "end_turn",
            ),
        )
        agent.run_turn("search the web")
        tool_result_msgs = [
            m for m in agent.session.messages
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert tool_result_msgs == []


class TestSession:
    def test_roundtrip_through_disk(self, tmp_path, monkeypatch):
        from src.models import config as config_mod

        monkeypatch.setattr(config_mod, "SESSION_DIR", tmp_path)
        monkeypatch.setattr("src.agent.session.SESSION_DIR", tmp_path)
        s = Session()
        s.add_user("hello")
        s.save()
        again = Session.load(s.id)
        assert again.messages[0]["content"] == "hello"
        assert again.title == "hello"

    def test_drop_system_messages_leaves_the_rest(self):
        s = Session()
        s.add_user("hi")
        s.add_system("context")
        s.add_user("again")
        s.drop_system_messages()
        assert [m["role"] for m in s.messages] == ["user", "user"]
