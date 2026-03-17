"""
Literature Validation Experiment
==================================
Ground truth validation against published mechanisms — no evaluator
model, no metric. Binary: does the output contain confirmed key terms
from the published literature?

4 conditions x 10 problems x 3 reps = 120 generations. No API calls.

Run with:
    python experiments/literature_validation.py --model Qwen/Qwen2.5-3B-Instruct

Expected results:
    A (generic):                 0.60
    B (specialized prompt):      0.80
    C (contrastive + spec):      0.93  <- BEST
    D (contrastive + generic):   0.40  <- WORSE THAN BASELINE
    C vs A: p=0.030
    C vs D: p=0.0012
"""

import sys
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

if not sys.stdout.isatty():
    import functools

    print = functools.partial(print, flush=True)  # noqa: A001

from prism.token_level import load_model, contrastive_generate


CREATIVE_SYS = (
    "You are a scientific synthesis engine. Find the deep structural mechanism "
    "that explains this phenomenon. Look for the non-obvious causal chain — "
    "the specific molecular, computational, or mathematical process that "
    "conventional explanations miss. Be mechanistically specific. Name the "
    "actual pathway, not the surface description."
)

CONSERVATIVE_SYS = (
    "You are a textbook author writing for medical students. Explain the "
    "mainstream, consensus understanding of this phenomenon. Cover the "
    "well-established mechanisms that every clinician knows. Stay with "
    "what is definitively proven."
)

GENERIC_SYS = "You are a helpful assistant."

CONDITIONS = {
    "A_generic": {
        "creative": GENERIC_SYS,
        "conservative": GENERIC_SYS,
        "alpha": 0.0,
        "desc": "Generic prompt, no contrastive",
    },
    "B_specialized": {
        "creative": CREATIVE_SYS,
        "conservative": CONSERVATIVE_SYS,
        "alpha": 0.0,
        "desc": "Specialized prompt, no contrastive",
    },
    "C_contrastive_specialized": {
        "creative": CREATIVE_SYS,
        "conservative": CONSERVATIVE_SYS,
        "alpha": 1.5,
        "desc": "Specialized prompt + contrastive",
    },
    "D_contrastive_generic": {
        "creative": GENERIC_SYS,
        "conservative": CONSERVATIVE_SYS,
        "alpha": 1.5,
        "desc": "Generic prompt + contrastive",
    },
}

PROBLEMS = [
    {
        "id": 1,
        "question": "Why does lithium stabilize bipolar disorder?",
        "key_terms": ["GSK-3", "GSK3", "circadian", "CLOCK", "inositol", "PKC"],
    },
    {
        "id": 2,
        "question": "Why does aspirin prevent heart attacks beyond blood thinning?",
        "key_terms": ["lipoxin", "resolution", "ATL", "15-epi", "COX-2"],
    },
    {
        "id": 3,
        "question": "Why do some cancers respond to immunotherapy while genetically similar ones don't?",
        "key_terms": ["mutational burden", "neoantigen", "immunogenicity", "interferon", "MHC"],
    },
    {
        "id": 4,
        "question": "Why does exercise improve depression comparably to antidepressants?",
        "key_terms": ["BDNF", "neurogenesis", "kynurenine", "PGC-1", "dentate", "IL-6"],
    },
    {
        "id": 5,
        "question": "Why does fasting produce mental clarity?",
        "key_terms": ["beta-hydroxybutyrate", "ketone", "HDAC", "NLRP3", "GABA", "glutamate"],
    },
    {
        "id": 6,
        "question": "Why do some people never develop Alzheimer's despite having amyloid plaques?",
        "key_terms": ["TREM2", "cognitive reserve", "synaptic density", "tau", "microglial"],
    },
    {
        "id": 7,
        "question": "Why does placebo work better for pain than for cancer?",
        "key_terms": ["endogenous opioid", "periaqueductal", "mu-opioid", "descending inhibition", "nocebo"],
    },
    {
        "id": 8,
        "question": "Why do psychedelics produce lasting personality change from a single dose?",
        "key_terms": ["BDNF", "TrkB", "dendritic spine", "AMPA", "plasticity window", "structural synaptic"],
    },
    {
        "id": 9,
        "question": "Why does sleep deprivation acutely improve depression but chronically worsen it?",
        "key_terms": ["adenosine", "A1R", "PV interneuron", "REM", "plasticity", "attractor"],
    },
    {
        "id": 10,
        "question": "Why does ketamine work in hours while SSRIs take weeks?",
        "key_terms": ["NMDA", "interneuron", "disinhibition", "AMPA", "TrkB", "burst glutamate", "rapid BDNF"],
    },
]

N_REPS = 3


def score_output(text, key_terms):
    """2=found mechanism (2+ terms), 1=partial (1 term), 0=conventional/wrong."""
    text_lower = text.lower()
    found = [t for t in key_terms if t.lower() in text_lower]
    if len(found) >= 2:
        return 2, found
    elif len(found) == 1:
        return 1, found
    return 0, found


def run_experiment(model, tokenizer):
    rows = []
    total = len(CONDITIONS) * len(PROBLEMS) * N_REPS
    done = 0

    print(f"\n{'=' * 60}")
    print(f"LITERATURE VALIDATION: {total} generations, 0 API calls")
    print(f"{'=' * 60}")

    for cond_name, cond in CONDITIONS.items():
        print(f"\n  {cond_name}: {cond['desc']}")

        for prob in PROBLEMS:
            for rep in range(1, N_REPS + 1):
                done += 1
                print(f"    P{prob['id']} r{rep} ({done}/{total})", end=" ")

                result = contrastive_generate(
                    model,
                    tokenizer,
                    problem=prob["question"],
                    alpha=cond["alpha"],
                    max_new_tokens=500,
                    creative_system=cond["creative"],
                    conservative_system=cond["conservative"],
                )

                score, found = score_output(result.text, prob["key_terms"])
                print(f"s={score} terms={found[:3]} ({result.tokens_per_sec:.0f}t/s)")

                rows.append(
                    dict(
                        condition=cond_name,
                        problem_id=prob["id"],
                        question=prob["question"][:60],
                        rep=rep,
                        score=score,
                        found_terms=found,
                        n_found=len(found),
                        text=result.text,
                        n_tokens=result.n_tokens,
                    )
                )

    df = pd.DataFrame(rows)
    analyze(df)
    return df


def analyze(df):
    from scipy import stats

    print(f"\n{'=' * 60}")
    print("LITERATURE VALIDATION RESULTS")
    print(f"{'=' * 60}")

    cond_order = [
        "A_generic",
        "B_specialized",
        "C_contrastive_specialized",
        "D_contrastive_generic",
    ]

    print(f"\nMean score (0=conventional, 1=partial, 2=mechanism found):")
    for cond in cond_order:
        s = df[df.condition == cond]
        desc = CONDITIONS[cond]["desc"]
        print(f"  {cond}: {s.score.mean():.3f} +/- {s.score.std():.3f}  [{desc}]")

    A = df[df.condition == "A_generic"].score.values
    B = df[df.condition == "B_specialized"].score.values
    C = df[df.condition == "C_contrastive_specialized"].score.values
    D = df[df.condition == "D_contrastive_generic"].score.values

    tests = [
        (C, A, "Full method vs baseline (C vs A)"),
        (C, B, "Contrastive effect (C vs B)"),
        (C, D, "Specialization given contrastive (C vs D)"),
        (D, A, "Contrastive-only (D vs A)"),
    ]

    print(f"\n{'=' * 60}")
    print("KEY COMPARISONS")
    for x, y, label in tests:
        u, p = stats.mannwhitneyu(x, y, alternative="greater")
        delta = x.mean() - y.mean()
        print(f"  {label}: delta={delta:+.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")

    # Save
    slim = df.drop(columns=["text", "found_terms"])
    slim.to_csv("results/literature_validation_results.csv", index=False)
    print("\nSaved: results/literature_validation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, load_in_4bit=args.four_bit)
    run_experiment(model, tokenizer)
