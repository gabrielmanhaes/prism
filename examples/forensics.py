"""
Forensic Analysis — Anomaly detection in text.

Uses the forensic cognitive mode (alpha=0.5) to surface patterns
that conventional analysis suppresses.
"""

from prism import TokenLevelPRISM, PRISMConfig, get_mode

prism = TokenLevelPRISM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    config=PRISMConfig(max_new_tokens=400),
)

mode = get_mode("forensic")

problem = """
Analyze this clinical trial report for anomalies:

A randomized controlled trial of Drug X for treatment-resistant depression
enrolled 200 patients. The primary endpoint (HAM-D reduction at 8 weeks)
showed significant improvement (p=0.03). However, the secondary endpoints
(quality of life, functional status) showed no significant change. Dropout
rate was 15% in the treatment arm and 8% in placebo. The per-protocol
analysis showed p=0.008 while the intent-to-treat analysis showed p=0.03.
"""

result = prism.generate(
    problem=problem,
    creative_system=mode.creative_system,
    conservative_system=mode.conservative_system,
    alpha=mode.optimal_alpha,
)

print(f"Mode: {mode.name} (alpha={mode.optimal_alpha})")
print(f"\n{result.text}")
