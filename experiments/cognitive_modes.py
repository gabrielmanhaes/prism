"""
Cognitive State Map Experiment
================================
Demonstrates that different cognitive tasks have different optimal
alpha values, consistent with the precision dynamics account.

Results from paper:
    synthesis:      optimal alpha=1.5
    depth:          optimal alpha=1.5
    adversarial:    optimal alpha=1.0
    contradiction:  optimal alpha=1.0
    forensic:       optimal alpha=0.5
    uncertainty:    optimal alpha=0.5
    temporal:       optimal alpha=0.5
    analogy:        optimal alpha=2.0

Run with:
    python experiments/cognitive_modes.py --model Qwen/Qwen2.5-3B-Instruct
    python experiments/cognitive_modes.py --mode synthesis --alpha_sweep
"""

import sys
import argparse
import json

import numpy as np
import pandas as pd

if not sys.stdout.isatty():
    import functools

    print = functools.partial(print, flush=True)  # noqa: A001

from prism.token_level import load_model, contrastive_generate
from prism.core.modes import MODES, get_mode


MODE_PROBLEMS = {
    "synthesis": [
        "Why does sleep deprivation paradoxically improve depression?",
        "How do antibiotic-resistant bacteria always defeat targeted treatments?",
    ],
    "forensic": [
        "Analyze this clinical trial: primary endpoint p=0.03 but all secondary endpoints non-significant, dropout 15% treatment vs 8% placebo.",
        "Examine this argument: 'AI will replace all jobs because it's getting exponentially better.'",
    ],
    "uncertainty": [
        "What do we actually know about consciousness?",
        "How well do we understand dark matter?",
    ],
    "contradiction": [
        "Find contradictions: 'We value diversity of thought. All team members must align with our core principles. Innovation requires challenging assumptions. Our methodology is proven and should not be questioned.'",
    ],
    "adversarial": [
        "What is the weakest point of the argument that increasing minimum wage always reduces employment?",
    ],
    "depth": [
        "What is the single most important mechanism explaining antibiotic resistance?",
    ],
    "temporal": [
        "What are the 10-year systemic consequences of widespread remote work?",
    ],
    "analogy": [
        "What structural pattern connects immune system function to distributed computing?",
    ],
}

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_REPS = 2


def run_mode_sweep(model, tokenizer, mode_name):
    """Sweep alpha for a single mode."""
    mode = get_mode(mode_name)
    problems = MODE_PROBLEMS.get(mode_name, MODE_PROBLEMS["synthesis"])
    rows = []

    print(f"\nMode: {mode.name} (predicted optimal: {mode.optimal_alpha})")

    for alpha in ALPHAS:
        for problem in problems:
            for rep in range(1, N_REPS + 1):
                print(f"  a={alpha:.2f} {problem[:40]}...", end=" ")
                result = contrastive_generate(
                    model,
                    tokenizer,
                    problem=problem,
                    alpha=alpha,
                    max_new_tokens=400,
                    creative_system=mode.creative_system,
                    conservative_system=mode.conservative_system,
                )
                print(f"{result.n_tokens}tok")
                rows.append(
                    dict(
                        mode=mode_name,
                        alpha=alpha,
                        problem=problem[:60],
                        rep=rep,
                        text=result.text,
                        n_tokens=result.n_tokens,
                    )
                )

    return rows


def run_all_modes(model, tokenizer):
    """Run alpha sweep for all modes."""
    all_rows = []
    for mode_name in MODE_PROBLEMS:
        all_rows.extend(run_mode_sweep(model, tokenizer, mode_name))

    df = pd.DataFrame(all_rows)
    df.drop(columns=["text"]).to_csv("results/cognitive_map_results.csv", index=False)
    print(f"\nSaved: results/cognitive_map_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", default=None, help="Run single mode (default: all)")
    parser.add_argument("--alpha_sweep", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.mode:
        run_mode_sweep(model, tokenizer, args.mode)
    else:
        run_all_modes(model, tokenizer)
