import os
from typing import Optional

from .base import BaseLLMProvider, GenerationConfig, GenerationResult


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider for open source models."""

    DEFAULT_MODEL = "mistral"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required: pip install requests")

        full_prompt = prompt
        if config.system_prompt:
            full_prompt = f"{config.system_prompt}\n\n{prompt}"

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            content=data.get("response", "").strip(),
            model=self.model,
            provider="ollama",
            tokens_used=data.get("eval_count"),
        )

    def is_available(self) -> bool:
        try:
            import requests

            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"
