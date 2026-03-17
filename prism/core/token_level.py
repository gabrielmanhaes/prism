"""
Token-level contrastive generation implementing cognitive state modulation.

At each token step, generates from two distributions:
- creative_dist: P(token | creative_system_prompt, context)
- conservative_dist: P(token | conservative_system_prompt, context)

Selects tokens via:
    score(t) = log P_creative(t) - alpha * log P_conservative(t)

subject to plausibility constraint:
    score(t) = -inf if P_creative(t) < threshold * max(P_creative)

This extends Li et al. 2023 (Contrastive Decoding) with cognitive state
framing rather than capability amplification.

Requires: torch, transformers
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PRISMConfig:
    """Configuration for token-level contrastive generation."""

    alpha: float = 1.5
    temperature: float = 1.0
    plausibility_threshold: float = 0.1
    max_new_tokens: int = 500
    device: str = "cuda"
    load_in_4bit: bool = False


@dataclass
class TokenLevelResult:
    """Result from token-level contrastive generation."""

    text: str
    n_tokens: int
    tokens_per_sec: float
    alpha: float
    problem: str


class TokenLevelPRISM:
    """
    Token-level contrastive generation implementing cognitive state modulation.

    At each token step, runs two forward passes through the same model
    with different system prompts, then combines the distributions:

        score(t) = log P_creative(t) - alpha * log P_conservative(t)

    alpha controls precision reduction depth:
        alpha=0.0  -> pure creative generation
        alpha=0.5  -> mild contrast (forensics, uncertainty)
        alpha=1.0  -> moderate contrast (adversarial, contradiction)
        alpha=1.5  -> strong contrast (scientific synthesis) [OPTIMAL]
        alpha=2.0  -> maximum contrast (analogy construction)
        alpha>2.0  -> incoherence threshold

    Uses KV caching so each generation step requires only two
    single-token forward passes after the initial prompt encoding.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[PRISMConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PRISMConfig()
        self.model.eval()

    def _build_prompt(self, system: str, user: str) -> str:
        """Build a chat prompt using the tokenizer's template."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{system}\n\n{user}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{system}\n\n{user}"

    @torch.no_grad()
    def generate(
        self,
        problem: str,
        creative_system: str,
        conservative_system: str,
        alpha: Optional[float] = None,
    ) -> TokenLevelResult:
        """
        Generate output using token-level contrastive mechanism.

        Args:
            problem: The input problem or question
            creative_system: System prompt inducing creative/low-P state
            conservative_system: System prompt inducing conventional/high-P state
            alpha: Contrastive strength (overrides config if provided)

        Returns:
            TokenLevelResult with generated text and metadata
        """
        alpha = alpha if alpha is not None else self.config.alpha
        device = next(self.model.parameters()).device

        creative_ids = self.tokenizer(
            self._build_prompt(creative_system, problem), return_tensors="pt"
        ).input_ids.to(device)
        conservative_ids = self.tokenizer(
            self._build_prompt(conservative_system, problem), return_tensors="pt"
        ).input_ids.to(device)

        # Pre-compute KV caches for both prompts
        c_out = self.model(creative_ids, use_cache=True)
        c_kv = c_out.past_key_values
        c_logits = c_out.logits[:, -1, :]

        k_out = self.model(conservative_ids, use_cache=True)
        k_kv = k_out.past_key_values
        k_logits = k_out.logits[:, -1, :]

        generated_ids = []
        log_thresh = math.log(self.config.plausibility_threshold)
        start = time.time()

        for _ in range(self.config.max_new_tokens):
            # Contrastive scoring
            c_lp = F.log_softmax(c_logits / self.config.temperature, dim=-1)
            k_lp = F.log_softmax(k_logits / self.config.temperature, dim=-1)
            score = c_lp - alpha * k_lp

            # Plausibility mask
            score = score.masked_fill(
                c_lp < (c_lp.max() + log_thresh), float("-inf")
            )

            probs = F.softmax(score, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tid = next_token.item()
            generated_ids.append(tid)

            if tid in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                break

            # Extend both KV caches with the sampled token
            next_input = next_token.view(1, 1)

            c_out = self.model(
                input_ids=next_input, past_key_values=c_kv, use_cache=True
            )
            c_kv = c_out.past_key_values
            c_logits = c_out.logits[:, -1, :]

            k_out = self.model(
                input_ids=next_input, past_key_values=k_kv, use_cache=True
            )
            k_kv = k_out.past_key_values
            k_logits = k_out.logits[:, -1, :]

        elapsed = time.time() - start
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return TokenLevelResult(
            text=text,
            n_tokens=len(generated_ids),
            tokens_per_sec=len(generated_ids) / elapsed if elapsed > 0 else 0,
            alpha=alpha,
            problem=problem,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: Optional[PRISMConfig] = None,
        **kwargs,
    ) -> "TokenLevelPRISM":
        """Load model and tokenizer from HuggingFace.

        Args:
            model_name: HuggingFace model ID (e.g. "mistralai/Mistral-7B-Instruct-v0.3")
            config: Optional PRISMConfig
            **kwargs: Passed to AutoModelForCausalLM.from_pretrained

        Returns:
            Initialized TokenLevelPRISM instance
        """
        config = config or PRISMConfig()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = dict(device_map="auto", **kwargs)

        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs.setdefault("torch_dtype", torch.bfloat16)

        # Try flash attention, fall back silently
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="flash_attention_2", **model_kwargs
            )
        except (ImportError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        return cls(model, tokenizer, config)
