"""
PRISM Quick Start — Scientific Synthesis

Finds cross-domain structural connections using token-level
contrastive generation.

Requirements:
    pip install prism-synthesis
    A local instruction-tuned model via HuggingFace
    Tested: Mistral-7B-Instruct, Qwen2.5-3B-Instruct
"""

from prism import TokenLevelPRISM, PRISMConfig, get_mode

# Load model (use your local model path or HuggingFace name)
prism = TokenLevelPRISM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    config=PRISMConfig(alpha=1.5),
)

# Get the synthesis cognitive mode
mode = get_mode("synthesis")

# Generate
problem = "Why does sleep deprivation paradoxically improve depression?"
result = prism.generate(
    problem=problem,
    creative_system=mode.creative_system,
    conservative_system=mode.conservative_system,
    alpha=mode.optimal_alpha,
)

print(f"Alpha: {result.alpha}")
print(f"Tokens: {result.n_tokens} ({result.tokens_per_sec:.0f} tok/s)")
print(f"\n{result.text}")
