"""
Scientific Synthesis — Sweep across alpha values.

Generates outputs for the same problem at different alpha values
to demonstrate the precision reduction effect.
"""

from prism import TokenLevelPRISM, PRISMConfig, get_mode

prism = TokenLevelPRISM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    config=PRISMConfig(max_new_tokens=400),
)

mode = get_mode("synthesis")
problem = "Why does sleep deprivation paradoxically improve depression?"

for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
    print(f"\n{'=' * 60}")
    print(f"Alpha = {alpha}")
    print("=" * 60)

    result = prism.generate(
        problem=problem,
        creative_system=mode.creative_system,
        conservative_system=mode.conservative_system,
        alpha=alpha,
    )
    print(result.text[:500])
