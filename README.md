# Atlas

A personal Claude agent that runs in your terminal. It reads and writes files in
directories you nominate, searches the web, runs shell commands behind an
approval gate, and remembers things about you across sessions.

Built on the Anthropic API directly, with no framework in between.

```
› what did I decide about the trimeffects port?

  memory? 1 match
Yarn mappings stopped at 1.21.11, so the 26.x port targets the unobfuscated
jar directly rather than waiting for mappings...
  $0.0184 this session
```

## Setup

```bash
cd atlas
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env      # then put your key in it
.venv/bin/python main.py
```

Get an API key at [console.anthropic.com](https://console.anthropic.com/settings/keys),
or run `ant auth login` and leave `ANTHROPIC_API_KEY` unset, and the SDK picks up the
profile automatically.

First launch downloads a ~80 MB local embedding model for memory. It happens once,
at startup, with a spinner.

## Usage

```bash
python main.py                      # interactive
python main.py -p "summarize ~/Desktop/notes.md"   # one-shot
python main.py --resume LAST        # continue the previous session
python main.py --thinking           # show the model's reasoning as it works
```

In the REPL:

| Command | |
|---|---|
| `/memory [query]` | list or search long-term memory |
| `/forget <id>` | delete a memory |
| `/sessions` | list saved sessions |
| `/thinking` | toggle reasoning display |
| `/cost` | tokens and spend for this session |
| `/clear` | fresh conversation, memory kept |
| `/exit` | quit |

## What it can do

### Files

`read_file`, `write_file`, `edit_file`, `list_dir`, `find_files`,
`search_text`. Every path is resolved and checked against `ATLAS_WORKSPACE_ROOTS`
before anything touches disk, so `..` traversal and symlinks pointing outside the
workspace are refused rather than followed.

### Web

`web_search` and `web_fetch` run server-side on Anthropic's
infrastructure. Nothing to install, and results are filtered before they reach the
context window.

### Shell

`run_command`. Destructive patterns (`rm -rf`, `sudo`, `curl | sh`,
`dd of=`, force-push, fork bombs) are blocked outright. Read-only commands are
allowlisted and run without interruption. Everything else prompts you for approval
before it runs.

### Memory

`remember`, `recall`, `forget`, backed by a local Chroma vector store.
Relevant memories are retrieved and injected automatically at the start of each
turn, so you don't have to ask. Memory is local; nothing is sent anywhere except as
context on your own requests.

## Configuration

Everything lives in `.env`. See `.env.example` for the full list.

| Variable | Default | |
|---|---|---|
| `ATLAS_MODEL` | `claude-opus-5` | any current Claude model id |
| `ATLAS_EFFORT` | `high` | `low` to `max`; how hard it thinks and works |
| `ATLAS_MAX_TOKENS` | `64000` | ceiling on thinking + response per turn |
| `ATLAS_WORKSPACE_ROOTS` | `~/Desktop` | colon-separated; the only reachable dirs |
| `ATLAS_SHELL_MODE` | `ask` | `ask`, `allowlist` (no prompts, deny unknown), `yolo` |
| `ATLAS_MEMORY_MIN_SCORE` | `0.15` | absolute relevance floor for recall |
| `ATLAS_MEMORY_REL_RATIO` | `0.6` | keep hits within this fraction of the best hit |

`ATLAS_EFFORT` is the main cost/quality dial. `high` is a good default; drop to
`medium` or `low` for routine work, raise to `xhigh` for hard multi-step tasks.

## Layout

```
main.py                  CLI entry point and REPL
src/
  agent/loop.py          the agent loop
  agent/session.py       conversation state and persistence
  agent/memory.py        vector memory
  tools/registry.py      tool registration and dispatch
  tools/{files,shell,memory_tools}.py
  tools/server.py        web search/fetch declarations
  models/config.py       every tunable, in one place
  prompts/system.py      the system prompt
  utils/{guardrails,logging,render}.py
tests/                   mirrors src/
data/                    memory store and saved sessions (gitignored)
logs/                    JSONL traces (gitignored)
```

## Design notes

### Why a manual loop, not the SDK tool runner

The runner is the usual
recommendation, but it doesn't resume a turn that stops with
`stop_reason: "pause_turn"`. It returns the paused turn as the final message instead. Since
this agent mixes client-side tools with server-side web search, which is exactly
what triggers a pause, that would surface as a silently truncated answer with no
error. The loop in `src/agent/loop.py` handles it explicitly, along with refusals
and token-ceiling truncation.

### Two-stage relevance filtering

Embedding scores for short queries sit in a
narrow band, and a *correct* hit for one query can score lower than an *incorrect*
hit for another: measured here, 0.231 versus 0.363. So a single absolute cutoff
either leaks noise or drops real matches. An absolute floor answers "is anything
relevant at all", then a relative floor keeps only what's competitive with the best
hit. Both are tunable; the defaults were calibrated against the local model and are
pinned by tests in `tests/test_memory.py`.

### Memory as a mid-conversation system message

Memory goes in as a mid-conversation system message, which keeps the cached
conversation prefix intact instead of rewriting the system prompt each turn. On
models that don't accept those, the agent falls back to folding memory into the user
turn automatically.

### Prompt caching

Caching is set on the system block, which holds the tools and system prompt and is
identical across every session, and on the conversation prefix. `/cost` shows how
many tokens came from cache.

### Traces

Every model request, tool call, and error is appended to
`logs/trace-YYYY-MM-DD.jsonl` with token counts and cost. Anything resembling a
credential is redacted before it's written.

## Tests

```bash
.venv/bin/python -m pytest
```

57 tests, no network or API key required. The guardrail tests cover path traversal,
symlink escape, and command classification; the loop tests cover pause-turn
resumption, refusal handling, and not returning tool results for server-side tools.

## Limitations

- Cost figures are computed from list prices in `src/models/config.py`. Update them
  if you switch model tiers.
- The shell allowlist is matched on the first word only, and commands containing
  shell operators always prompt. It is a speed bump, not a sandbox, so don't set
  `ATLAS_SHELL_MODE=yolo` and walk away.
- Long conversations aren't compacted. The 1M-token context window means this is
  unlikely to bite in a personal session, but a very long one will eventually
  exhaust it.
