from types import SimpleNamespace

import pytest

from src.models.config import DEFAULT_MODEL, PRICING, Config


def usage(fresh=0, out=0, read=0, write=0):
    return SimpleNamespace(
        input_tokens=fresh,
        output_tokens=out,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=write,
    )


def test_default_model_has_a_price():
    assert DEFAULT_MODEL in PRICING


def test_opus_5_5_cost_uses_its_own_cache_read_price():
    cfg = Config(model="claude-opus-5-5")
    # $4 in, $20 out, $0.20 cache read, cache write at 1.25x input = $5.
    m = 1_000_000
    assert cfg.cost(usage(fresh=m, out=m, read=m, write=m)) == pytest.approx(4 + 20 + 0.20 + 5)


def test_haiku_5_5_switches_to_long_prompt_rates_past_100k():
    cfg = Config(model="claude-haiku-5-5")
    assert cfg.cost(usage(fresh=50_000)) == pytest.approx(50_000 * 0.10 / 1e6)
    assert cfg.cost(usage(fresh=200_000)) == pytest.approx(200_000 * 0.50 / 1e6)
    # Cached tokens count toward the prompt size that picks the tier.
    assert cfg.price(prompt_tokens=150_000)[0] == 0.50


def test_unknown_model_costs_zero_and_says_so():
    cfg = Config(model="claude-imaginary-9")
    assert cfg.has_price() is False
    assert cfg.cost(usage(fresh=1_000, out=1_000)) == 0.0
