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

# Published list prices, USD per million tokens. Used only for the local cost
# readout -- update alongside ATLAS_MODEL if you switch tiers.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
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
    "ls,cat,head,tail,wc,grep,rg,find,file,stat,du,df,"
    "git,python3,python,pip,pytest,node,npm,jq,which,echo,pwd,date"
)


def _allowlist() -> set[str]:
    raw = os.getenv("ATLAS_SHELL_ALLOWLIST") or DEFAULT_SHELL_ALLOWLIST
    return {c.strip() for c in raw.split(",") if c.strip()}


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.getenv("ATLAS_MODEL", "claude-opus-5"))
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

    def price(self) -> tuple[float, float]:
        """(input, output) USD per million tokens for the configured model."""
        return PRICING.get(self.model, (0.0, 0.0))

    def cost(self, usage) -> float:
        """Approximate USD for one response's usage object.

        Cache writes bill at ~1.25x input, cache reads at ~0.1x.
        """
        pin, pout = self.price()
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        fresh = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        return (
            fresh * pin + write * pin * 1.25 + read * pin * 0.1 + out * pout
        ) / 1_000_000


CONFIG = Config()

for _d in (DATA_DIR, LOG_DIR, SESSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)
