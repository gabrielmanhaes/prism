import os
from typing import Optional

from .base import BaseLLMProvider, GenerationConfig, GenerationResult


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required: pip install prism-synthesis[openai]")
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        client = self._get_client()

        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        return GenerationResult(
            content=response.choices[0].message.content,
            model=self.model,
            provider="openai",
            tokens_used=response.usage.total_tokens if response.usage else None,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def name(self) -> str:
        return f"openai/{self.model}"
