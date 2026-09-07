import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def isolate_side_effects(tmp_path_factory, monkeypatch):
    """Keep tests out of the real logs/ and data/ directories."""
    import src.utils.logging as logging_mod

    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path_factory.mktemp("logs"))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the guardrails at a throwaway directory."""
    from src.models import config as config_mod

    monkeypatch.setattr(config_mod.CONFIG, "workspace_roots", [tmp_path.resolve()], raising=False)
    return tmp_path
