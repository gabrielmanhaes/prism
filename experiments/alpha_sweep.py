"""
Alpha Sweep Experiment
========================
Sweeps alpha from 0.0 to 2.5 in 0.25 increments for synthesis tasks.
Measures PSS at each point to find optimal contrastive strength.

Run with:
    python experiments/alpha_sweep.py --model Qwen/Qwen2.5-3B-Instruct

Expected: PSS peaks at alpha=1.5, degrades at alpha>2.0
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
from prism.core.modes import get_mode


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

PROBLEMS = [
    ("Why does sleep deprivation paradoxically improve depression?", "neuroscience", "information theory"),
    ("How do antibiotic-resistant bacteria always defeat targeted treatments?", "microbiology", "game theory"),
    ("Why do antidepressants take 2-4 weeks to work?", "pharmacology", "dynamical systems"),
    ("Why does ketamine produce rapid antidepressant effects lasting weeks?", "neuropharmacology", "control theory"),
    ("Why do psychedelics produce lasting personality change from a single dose?", "neuroscience", "topology"),
]

N_REPS = 3


def run_experiment(model, tokenizer):
    mode = get_mode("synthesis")
    rows = []
    total = len(ALPHAS) * len(PROBLEMS) * N_REPS
    done = 0

    for alpha in ALPHAS:
        print(f"\n{'=' * 60}")
        print(f"Alpha = {alpha}")

        for problem, domain_a, domain_b in PROBLEMS:
            for rep in range(1, N_REPS + 1):
                done += 1
                print(f"  ({done}/{total}) a={alpha} {problem[:40]}...", end=" ")

                result = contrastive_generate(
                    model,
                    tokenizer,
                    problem=problem,
                    domain_a=domain_a,
                    domain_b=domain_b,
                    alpha=alpha,
                    max_new_tokens=500,
                    creative_system=mode.creative_system,
                    conservative_system=mode.conservative_system,
                )

                print(f"{result.n_tokens}tok ({result.tokens_per_sec:.0f}t/s)")

                rows.append(
                    dict(
                        alpha=alpha,
                        problem=problem,
                        domain_a=domain_a,
                        domain_b=domain_b,
                        rep=rep,
                        text=result.text,
                        n_tokens=result.n_tokens,
                        tokens_per_sec=result.tokens_per_sec,
                    )
                )

    df = pd.DataFrame(rows)
    df.drop(columns=["text"]).to_csv("results/alpha_sweep_results.csv", index=False)
    print(f"\nSaved: results/alpha_sweep_results.csv")

    with open("results/alpha_sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("Saved: results/alpha_sweep_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, load_in_4bit=args.four_bit)
    run_experiment(model, tokenizer)
