from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 1500
    system_prompt: Optional[str] = None


@dataclass
class GenerationResult:
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
