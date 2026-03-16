"""
PRISM — Precision Reduction for Isomorphic Synthesis and Mapping

Find structural equivalences between distant scientific domains.

Quick start:
    from prism import PRISM

    # Auto-detect available provider
    engine = PRISM.from_env()

    result = engine.synthesize(
        problem="Why does sleep deprivation paradoxically improve depression?",
        domain_a="neuroscience",
        domain_b="information theory"
    )
    print(result.synthesis)
    print(f"PSS: {result.pss.pss:.4f}")
"""

from .engine import PRISM, PRISMConfig, PRISMResult
from .pss import PSS, PSSResult
from .providers.base import BaseLLMProvider, GenerationConfig, GenerationResult
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .providers.ollama import OllamaProvider


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
    "PRISM",
    "PRISMConfig",
    "PRISMResult",
    "PSS",
    "PSSResult",
    "BaseLLMProvider",
    "GenerationConfig",
    "GenerationResult",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
]
