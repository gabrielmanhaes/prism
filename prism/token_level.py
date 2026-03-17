"""
Token-level contrastive plasticity engine.

Applies the creative-conservative contrast at every token generation step
using KV-cached dual forward passes on a local GPU model.

Requires: torch, transformers
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TokenLevelResult:
    text: str
    n_tokens: int
    tokens_per_sec: float
    alpha: Optional[float]
    method: str  # "token_level" or "response_level"
    problem: str
    domain_a: str
    domain_b: str


# ── Prompts (tuned for local instruct models) ────────────────────────────────

CREATIVE_TEMPLATE = (
    "You are a scientific synthesis engine. Find the deep structural "
    "connection between {domain_a} and {domain_b} that illuminates the "
    "given problem. Look for mathematical equivalences, formal "
    "isomorphisms, and organizing principles that transfer entire "
    "problem-solving frameworks between fields. Be structurally "
    "specific — name the formal tools that transfer."
)

CONSERVATIVE_TEMPLATE = (
    "You are a textbook. Provide a conventional, standard, "
    "well-established scientific explanation of the given problem "
    "from the perspective of {domain_a}. Stick to mainstream "
    "interpretations. Do not speculate or introduce novel connections."
)

DELTA_SYSTEM = (
    "Extract only the ideas present in the creative response that are "
    "absent from the conventional response. State directly, no "
    "meta-commentary."
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, load_in_4bit: bool = False, device: str = "cuda"):
    """Load a HuggingFace model for token-level generation."""
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto")

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    # Try flash attention, fall back silently
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **kwargs
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model.eval()

    if torch.cuda.is_available():
        print(f"Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    else:
        print("Loaded (CPU).")

    return model, tokenizer


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat prompt using the tokenizer's template."""
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system}\n\n{user}"


# ── Token-level contrastive generation ────────────────────────────────────────

@torch.no_grad()
def contrastive_generate(
    model,
    tokenizer,
    problem: str,
    domain_a: str = "",
    domain_b: str = "",
    alpha: float = 0.5,
    max_new_tokens: int = 600,
    temperature: float = 1.0,
    plausibility_threshold: float = 0.1,
    creative_system: Optional[str] = None,
    conservative_system: Optional[str] = None,
) -> TokenLevelResult:
    """
    Token-level contrastive plasticity with KV caching.

    At each step:
      creative_logits   = forward(creative_prompt + generated_so_far)
      conservative_logits = forward(conservative_prompt + generated_so_far)
      score = log_softmax(creative) - alpha * log_softmax(conservative)
      next_token = sample(softmax(score))

    Both prompts share the same generated suffix. Two KV caches are
    maintained so each step requires only two single-token forward passes.
    """
    device = next(model.parameters()).device

    if creative_system is not None:
        creative_sys = creative_system
    else:
        creative_sys = CREATIVE_TEMPLATE.format(domain_a=domain_a, domain_b=domain_b)
    if conservative_system is not None:
        conservative_sys = conservative_system
    else:
        conservative_sys = CONSERVATIVE_TEMPLATE.format(domain_a=domain_a)

    creative_ids = tokenizer(
        build_prompt(tokenizer, creative_sys, problem), return_tensors="pt"
    ).input_ids.to(device)
    conservative_ids = tokenizer(
        build_prompt(tokenizer, conservative_sys, problem), return_tensors="pt"
    ).input_ids.to(device)

    # Pre-compute KV caches for both prompts
    c_out = model(creative_ids, use_cache=True)
    c_kv = c_out.past_key_values
    c_logits = c_out.logits[:, -1, :]

    k_out = model(conservative_ids, use_cache=True)
    k_kv = k_out.past_key_values
    k_logits = k_out.logits[:, -1, :]

    generated_ids = []
    log_thresh = math.log(plausibility_threshold)
    start = time.time()

    for step in range(max_new_tokens):
        # Contrastive scoring
        c_lp = F.log_softmax(c_logits / temperature, dim=-1)
        k_lp = F.log_softmax(k_logits / temperature, dim=-1)
        score = c_lp - alpha * k_lp

        # Plausibility mask — only tokens the creative model finds plausible
        score = score.masked_fill(c_lp < (c_lp.max() + log_thresh), float("-inf"))

        probs = F.softmax(score, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tid = next_token.item()
        generated_ids.append(tid)

        if tid in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break

        # Extend both KV caches with the sampled token
        next_input = next_token.view(1, 1)

        c_out = model(input_ids=next_input, past_key_values=c_kv, use_cache=True)
        c_kv = c_out.past_key_values
        c_logits = c_out.logits[:, -1, :]

        k_out = model(input_ids=next_input, past_key_values=k_kv, use_cache=True)
        k_kv = k_out.past_key_values
        k_logits = k_out.logits[:, -1, :]

        if step > 0 and step % 100 == 0:
            print(f"  step {step}/{max_new_tokens} — {step / (time.time() - start):.0f} tok/s")

    elapsed = time.time() - start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return TokenLevelResult(
        text=text,
        n_tokens=len(generated_ids),
        tokens_per_sec=len(generated_ids) / elapsed if elapsed > 0 else 0,
        alpha=alpha,
        method="token_level",
        problem=problem,
        domain_a=domain_a,
        domain_b=domain_b,
    )


# ── Response-level generation (same local model, for fair comparison) ─────────

@torch.no_grad()
def response_level_generate(
    model,
    tokenizer,
    problem: str,
    domain_a: str,
    domain_b: str,
    creative_temperature: float = 1.0,
    max_new_tokens: int = 600,
) -> TokenLevelResult:
    """Standard 3-stage creative → conservative → delta on the same local model."""
    device = next(model.parameters()).device

    creative_sys = CREATIVE_TEMPLATE.format(domain_a=domain_a, domain_b=domain_b)
    conservative_sys = CONSERVATIVE_TEMPLATE.format(domain_a=domain_a)

    def _gen(system: str, user: str, temp: float) -> str:
        prompt = build_prompt(tokenizer, system, user)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temp, 0.01),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    start = time.time()
    creative = _gen(creative_sys, problem, creative_temperature)
    conservative = _gen(conservative_sys, problem, 0.2)

    delta_prompt = (
        f"CREATIVE RESPONSE:\n{creative}\n\n"
        f"CONVENTIONAL RESPONSE:\n{conservative}\n\n"
        f"Extract only the ideas present in the creative response "
        f"but absent from the conventional response."
    )
    delta = _gen(DELTA_SYSTEM, delta_prompt, 0.3)
    elapsed = time.time() - start

    return TokenLevelResult(
        text=delta,
        n_tokens=len(tokenizer.encode(delta)),
        tokens_per_sec=0,
        alpha=None,
        method="response_level",
        problem=problem,
        domain_a=domain_a,
        domain_b=domain_b,
    )
