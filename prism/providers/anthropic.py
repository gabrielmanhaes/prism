import os
from typing import Optional

from .base import BaseLLMProvider, GenerationConfig, GenerationResult


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install prism-synthesis[anthropic]")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if config.system_prompt:
            kwargs["system"] = config.system_prompt

        response = client.messages.create(**kwargs)

        return GenerationResult(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"
