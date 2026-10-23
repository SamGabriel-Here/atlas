"""Single source of truth for model ids and every tunable knob.

Everything here is overridable by environment variable so that churn-prone
values (model ids, effort, limits) never get scattered across call sites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
SESSION_DIR = DATA_DIR / "sessions"
MEMORY_DIR = DATA_DIR / "memory"

DEFAULT_MODEL = "claude-opus-5-5"

# Published list prices, USD per million tokens: (input, output, cache read).
# Used only for the local cost readout -- update alongside ATLAS_MODEL if you
# switch tiers. Cache reads are listed per model because the ratio to input
# varies (0.05x on Opus 5.5, 0.025x on Fable 5.1, 0.1x on most others).
PRICING: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00, 1.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}

# Models that bill a higher rate once the prompt passes a size threshold:
# (threshold in prompt tokens, input, output, cache read) above it.
# ponytail: Haiku 5.5's long-prompt cache-read rate is assumed at 0.1x input;
# replace it when the published figure is known.
LONG_PROMPT_PRICING: dict[str, tuple[int, float, float, float]] = {
    "claude-haiku-5-5": (100_000, 0.50, 2.50, 0.05),
}

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _roots() -> list[Path]:
    raw = os.getenv("ATLAS_WORKSPACE_ROOTS") or "~/Desktop"
    out = []
    for part in raw.split(":"):
        part = part.strip()
        if part:
            out.append(Path(part).expanduser().resolve())
    return out or [Path.home() / "Desktop"]


# Read-only or clearly-scoped commands that never need a confirmation prompt.
# Defaults live here, not only in .env.example, so a fresh clone with no .env
# still behaves sensibly instead of prompting for `ls`.
DEFAULT_SHELL_ALLOWLIST = (
    "ls,cat,head,tail,wc,grep,rg,find,file,stat,du,df,git,jq,which,echo,pwd,date"
)


def _allowlist() -> set[str]:
    raw = os.getenv("ATLAS_SHELL_ALLOWLIST") or DEFAULT_SHELL_ALLOWLIST
    return {c.strip() for c in raw.split(",") if c.strip()}


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.getenv("ATLAS_MODEL", DEFAULT_MODEL))
    effort: str = field(default_factory=lambda: os.getenv("ATLAS_EFFORT", "high"))
    max_tokens: int = field(default_factory=lambda: _env_int("ATLAS_MAX_TOKENS", 64000))

    workspace_roots: list[Path] = field(default_factory=_roots)
    shell_mode: str = field(default_factory=lambda: os.getenv("ATLAS_SHELL_MODE", "ask"))
    shell_allowlist: set[str] = field(default_factory=_allowlist)

    memory_enabled: bool = field(default_factory=lambda: _env_bool("ATLAS_MEMORY_ENABLED", True))
    memory_top_k: int = field(default_factory=lambda: _env_int("ATLAS_MEMORY_TOP_K", 5))
    # Absolute floor: below this, nothing is relevant enough to inject at all.
    memory_min_score: float = field(
        default_factory=lambda: _env_float("ATLAS_MEMORY_MIN_SCORE", 0.15)
    )
    # Relative floor: keep only hits within this fraction of the best hit's score.
    memory_rel_ratio: float = field(
        default_factory=lambda: _env_float("ATLAS_MEMORY_REL_RATIO", 0.6)
    )

    def has_price(self) -> bool:
        return self.model in PRICING

    def price(self, prompt_tokens: int = 0) -> tuple[float, float, float]:
        """(input, output, cache read) USD per million tokens for the configured model.

        Unknown models price at zero; check has_price() to warn about that.
        """
        tier = LONG_PROMPT_PRICING.get(self.model)
        if tier and prompt_tokens > tier[0]:
            return tier[1:]
        return PRICING.get(self.model, (0.0, 0.0, 0.0))

    def cost(self, usage) -> float:
        """Approximate USD for one response's usage object.

        Cache writes bill at 1.25x input (5-minute TTL); cache reads use the
        model's own cache-read price.
        """
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        fresh = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        pin, pout, pread = self.price(prompt_tokens=fresh + read + write)
        return (fresh * pin + write * pin * 1.25 + read * pread + out * pout) / 1_000_000


CONFIG = Config()

for _d in (DATA_DIR, LOG_DIR, SESSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)
