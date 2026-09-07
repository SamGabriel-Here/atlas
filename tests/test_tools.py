import pytest

from src.tools import files, memory_tools, shell  # noqa: F401 - registers tools
from src.tools.registry import ToolContext, client_tool_definitions, dispatch


@pytest.fixture
def ctx(workspace):
    return ToolContext(memory=None, approve=lambda command, reason: False, cwd=str(workspace))


class TestToolDefinitions:
    def test_every_tool_has_a_usable_schema(self):
        for defn in client_tool_definitions():
            assert defn["name"]
            # Descriptions must say when to call the tool, not just what it does.
            assert len(defn["description"]) > 80, defn["name"]
            schema = defn["input_schema"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            for prop in schema["properties"].values():
                assert prop.get("description"), defn["name"]
                assert "_required" not in prop, "schema helper leaked its marker key"


class TestFileTools:
    def test_read_file_returns_numbered_lines(self, ctx, workspace):
        p = workspace / "a.txt"
        p.write_text("one\ntwo\nthree\n")
        out = files.read_file(ctx=ctx, path=str(p))
        assert "1\tone" in out and "3\tthree" in out

    def test_read_file_respects_line_range(self, ctx, workspace):
        p = workspace / "a.txt"
        p.write_text("\n".join(str(n) for n in range(1, 21)))
        out = files.read_file(ctx=ctx, path=str(p), start_line=5, end_line=7)
        assert "lines 5-7 of 20" in out
        assert "\t4" not in out

    def test_write_then_read_roundtrip(self, ctx, workspace):
        target = workspace / "sub" / "new.md"
        files.write_file(ctx=ctx, path=str(target), content="hello")
        assert target.read_text() == "hello"

    def test_edit_file_replaces_unique_string(self, ctx, workspace):
        p = workspace / "c.py"
        p.write_text("x = 1\ny = 2\n")
        files.edit_file(ctx=ctx, path=str(p), old_string="y = 2", new_string="y = 3")
        assert p.read_text() == "x = 1\ny = 3\n"

    def test_edit_file_refuses_ambiguous_match(self, ctx, workspace):
        p = workspace / "d.py"
        p.write_text("dup\ndup\n")
        out = files.edit_file(ctx=ctx, path=str(p), old_string="dup", new_string="x")
        assert "appears 2 times" in out
        assert p.read_text() == "dup\ndup\n", "file must be untouched on an ambiguous edit"

    def test_edit_file_reports_missing_match(self, ctx, workspace):
        p = workspace / "e.py"
        p.write_text("abc")
        assert "not found" in files.edit_file(ctx=ctx, path=str(p), old_string="zzz", new_string="q")

    def test_search_text_finds_matches(self, ctx, workspace):
        (workspace / "f.py").write_text("def target():\n    pass\n")
        out = files.search_text(ctx=ctx, pattern=r"def target", path=str(workspace))
        assert "f.py:1" in out

    def test_search_text_reports_bad_regex(self, ctx, workspace):
        assert "Invalid regular expression" in files.search_text(
            ctx=ctx, pattern="[unclosed", path=str(workspace)
        )

    def test_find_files_matches_glob(self, ctx, workspace):
        (workspace / "g.py").write_text("")
        (workspace / "h.md").write_text("")
        out = files.find_files(ctx=ctx, pattern="*.py", path=str(workspace))
        assert "g.py" in out and "h.md" not in out

    def test_binary_file_is_not_read_as_text(self, ctx, workspace):
        p = workspace / "blob.bin"
        p.write_bytes(b"\x00\x01\x02binary")
        assert "binary file" in files.read_file(ctx=ctx, path=str(p))


class TestShellTool:
    def test_runs_allowlisted_command(self, ctx, workspace):
        (workspace / "marker.txt").write_text("")
        out = shell.run_command(ctx=ctx, command="ls", cwd=str(workspace))
        assert "marker.txt" in out

    def test_declined_command_returns_a_result_not_an_error(self, ctx, workspace):
        out = shell.run_command(ctx=ctx, command="brew install cowsay", cwd=str(workspace))
        assert "declined" in out.lower()

    def test_destructive_command_is_blocked_outright(self, ctx, workspace):
        from src.utils.guardrails import GuardrailError

        with pytest.raises(GuardrailError):
            shell.run_command(ctx=ctx, command="sudo rm -rf /", cwd=str(workspace))

    def test_nonzero_exit_is_reported_not_raised(self, ctx, workspace):
        out = shell.run_command(ctx=ctx, command="ls /definitely/not/here", cwd=str(workspace))
        assert "exit" in out


class TestDispatch:
    def test_unknown_tool_reports_cleanly(self, ctx):
        text, ok = dispatch("no_such_tool", {}, ctx)
        assert ok is False and "Unknown tool" in text

    def test_exception_becomes_an_error_result(self, ctx):
        text, ok = dispatch("read_file", {"path": "/etc/shadow"}, ctx)
        assert ok is False and "workspace" in text

    def test_bad_arguments_are_reported(self, ctx):
        text, ok = dispatch("read_file", {"wrong_arg": 1}, ctx)
        assert ok is False and "Bad arguments" in text
