from unittest.mock import MagicMock, patch

from prism.engine import PRISM, PRISMConfig
from prism.providers.base import GenerationResult


def _mock_provider(content="mock response"):
    provider = MagicMock()
    provider.generate.return_value = GenerationResult(
        content=content, model="mock", provider="mock"
    )
    provider.is_available.return_value = True
    provider.name = "mock/model"
    return provider


def test_prism_init():
    """PRISM should initialize with generator only."""
    gen = _mock_provider()
    engine = PRISM(generator=gen)
    assert engine.evaluator is gen
    assert engine.config.min_pss == 0.04


def test_prism_config():
    """Custom config should be applied."""
    config = PRISMConfig(min_pss=0.1, max_attempts=3, verbose=True)
    engine = PRISM(generator=_mock_provider(), config=config)
    assert engine.config.min_pss == 0.1
    assert engine.config.max_attempts == 3


def test_creative_pass():
    """Creative pass should call generator with high temperature."""
    gen = _mock_provider("creative output")
    engine = PRISM(generator=gen)
    result = engine._creative_pass("test problem", "domain_a", "domain_b")
    assert result == "creative output"
    assert gen.generate.called


def test_conservative_pass():
    """Conservative pass should call generator with low temperature."""
    gen = _mock_provider("conventional output")
    engine = PRISM(generator=gen)
    result = engine._conservative_pass("test problem", "domain_a")
    assert result == "conventional output"
    call_args = gen.generate.call_args
    assert call_args[0][1].temperature == 0.2


def test_delta_extraction():
    """Delta extraction should call generator."""
    gen = _mock_provider("delta content")
    engine = PRISM(generator=gen)
    result = engine._delta_extraction("creative", "conservative")
    assert result == "delta content"


def test_prism_default_domains():
    """Default domain list should exist and be non-empty."""
    from prism.domains import DEFAULT_DOMAINS

    assert len(DEFAULT_DOMAINS) >= 10
    assert "information theory" in DEFAULT_DOMAINS
