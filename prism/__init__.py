"""
PRISM — Precision Reduction for Isomorphic Synthesis and Mapping

Token-level contrastive generation that modulates the cognitive state
of language models — not just what they generate, but how they evaluate.

Quick start (token-level, local GPU):
    from prism import TokenLevelPRISM, PRISMConfig, get_mode

    prism = TokenLevelPRISM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        config=PRISMConfig(alpha=1.5),
    )
    mode = get_mode("synthesis")
    output = prism.generate(
        problem="Why does sleep deprivation paradoxically improve depression?",
        creative_system=mode.creative_system,
        conservative_system=mode.conservative_system,
    )
    print(output.text)

Quick start (response-level, API):
    from prism import PRISM

    engine = PRISM.from_env()
    result = engine.synthesize(
        problem="Why does sleep deprivation paradoxically improve depression?",
        domain_a="neuroscience",
        domain_b="information theory",
    )
    print(result.synthesis)
"""

# Response-level API engine
from .engine import PRISM, PRISMConfig as ResponseLevelConfig, PRISMResult
from .pss import PSS, PSSResult

# Providers
from .providers.base import BaseLLMProvider, GenerationConfig, GenerationResult
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .providers.ollama import OllamaProvider

# Core: token-level engine and cognitive modes
from .core.token_level import (
    TokenLevelPRISM,
    PRISMConfig,
    TokenLevelResult,
)
from .core.modes import CognitiveMode, MODES, get_mode, list_modes


def _auto_provider() -> BaseLLMProvider:
    """Auto-detect available LLM provider from environment."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    elif os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    else:
        provider = OllamaProvider()
        if provider.is_available():
            return provider
        raise EnvironmentError(
            "No LLM provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
            "or install and run Ollama locally."
        )


@classmethod  # type: ignore[misc]
def _from_env(cls, config=None):
    """Create PRISM instance using auto-detected provider from environment."""
    provider = _auto_provider()
    return cls(generator=provider, config=config)


PRISM.from_env = _from_env  # type: ignore[attr-defined]

__version__ = "0.1.0"
__all__ = [
    # Token-level (primary)
    "TokenLevelPRISM",
    "PRISMConfig",
    "TokenLevelResult",
    # Cognitive modes
    "CognitiveMode",
    "MODES",
    "get_mode",
    "list_modes",
    # Response-level API engine
    "PRISM",
    "ResponseLevelConfig",
    "PRISMResult",
    # Evaluation
    "PSS",
    "PSSResult",
    # Providers
    "BaseLLMProvider",
    "GenerationConfig",
    "GenerationResult",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
]
