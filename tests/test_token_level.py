"""Tests for token-level PRISM config and result types.

Note: Testing actual generation requires a GPU and model.
These tests validate the config, types, and class structure.
"""

from prism.core.token_level import PRISMConfig, TokenLevelResult, TokenLevelPRISM


def test_config_defaults():
    config = PRISMConfig()
    assert config.alpha == 1.5
    assert config.temperature == 1.0
    assert config.plausibility_threshold == 0.1
    assert config.max_new_tokens == 500


def test_config_custom():
    config = PRISMConfig(alpha=0.5, temperature=0.8, max_new_tokens=200)
    assert config.alpha == 0.5
    assert config.temperature == 0.8
    assert config.max_new_tokens == 200


def test_result_dataclass():
    result = TokenLevelResult(
        text="test output",
        n_tokens=10,
        tokens_per_sec=50.0,
        alpha=1.5,
        problem="test problem",
    )
    assert result.text == "test output"
    assert result.alpha == 1.5


def test_prism_class_has_required_methods():
    assert hasattr(TokenLevelPRISM, "generate")
    assert hasattr(TokenLevelPRISM, "from_pretrained")
