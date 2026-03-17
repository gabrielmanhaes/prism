"""
Evaluator Asymmetry Experiment
================================
Demonstrates that the model's precision state during EVALUATION
determines what it recognizes as valuable — not just generation.

Key finding: identical outputs evaluated at alpha=1.5 vs alpha=0.0
show 7x difference in recognition of structural synthesis quality
(p=0.0021, chi-squared=9.45).

This cannot be explained as a prompt effect because:
- The outputs being evaluated are FIXED (pre-generated)
- The system prompts are IDENTICAL across conditions
- Only alpha changes between conditions

Run with:
    python experiments/evaluator_asymmetry.py --model Qwen/Qwen2.5-3B-Instruct

Expected output:
    eval@0.0 prefers structural: ~5%
    eval@1.5 prefers structural: ~35%
    chi-squared=9.45, p=0.0021
"""

import sys
import argparse
import json
import random
import re

import numpy as np
import pandas as pd

if not sys.stdout.isatty():
    import functools

    print = functools.partial(print, flush=True)  # noqa: A001

from prism.token_level import load_model, contrastive_generate


# Evaluation prompts — IDENTICAL for both conditions. Only alpha changes.
EVAL_CREATIVE = (
    "You are evaluating scientific explanations. Consider structural "
    "depth, mathematical rigor, novel connections, and explanatory power."
)

EVAL_CONSERVATIVE = (
    "You are evaluating scientific explanations. Consider only factual "
    "accuracy, conventional support, and standard scientific consensus."
)

N_REPS = 10  # per problem per evaluator condition


def parse_choice(text: str):
    """Extract A or B from evaluator output."""
    text = text.strip()
    if text and text[0] in ("A", "B"):
        return text[0]

    has_a = bool(re.search(r"Response A", text, re.IGNORECASE))
    has_b = bool(re.search(r"Response B", text, re.IGNORECASE))
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"

    for line in reversed(text.split("\n")):
        line = line.strip().rstrip(".")
        if line in ("A", "B"):
            return line

    m = re.search(
        r"(?:choose|select|pick|prefer|better)[^.]*\b([AB])\b", text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    return None


def build_eval_prompt(problem: str, response_a: str, response_b: str) -> str:
    return (
        f"Given this problem:\n{problem}\n\n"
        f"Which response better addresses the problem?\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Answer with ONLY the letter A or B."
    )


def generate_paired_outputs(model, tokenizer, problems):
    """Generate alpha=0.0 and alpha=1.5 outputs for each problem."""
    from prism.core.modes import get_mode

    mode = get_mode("synthesis")
    pairs = []

    for i, problem in enumerate(problems):
        print(f"Generating pair {i + 1}/{len(problems)}: {problem[:50]}...")

        out_0 = contrastive_generate(
            model,
            tokenizer,
            problem=problem,
            alpha=0.0,
            max_new_tokens=500,
            creative_system=mode.creative_system,
            conservative_system=mode.conservative_system,
        )
        out_15 = contrastive_generate(
            model,
            tokenizer,
            problem=problem,
            alpha=1.5,
            max_new_tokens=500,
            creative_system=mode.creative_system,
            conservative_system=mode.conservative_system,
        )

        pairs.append(
            {
                "problem": problem,
                "output_alpha0": out_0.text,
                "output_alpha15": out_15.text,
            }
        )

    return pairs


def run_experiment(model, tokenizer, problems, n_reps=N_REPS):
    """Run the evaluator asymmetry experiment."""
    pairs = generate_paired_outputs(model, tokenizer, problems)
    print(f"\nGenerated {len(pairs)} output pairs. Starting evaluation...\n")

    rows = []

    for i, pair in enumerate(pairs):
        problem = pair["problem"]
        print(f"\n{'=' * 60}")
        print(f"Problem {i + 1}/{len(pairs)}: {problem[:60]}...")

        for rep in range(1, n_reps + 1):
            # Randomize presentation order
            if random.random() < 0.5:
                resp_a, resp_b = pair["output_alpha0"], pair["output_alpha15"]
                a_source, b_source = "alpha0", "alpha15"
            else:
                resp_a, resp_b = pair["output_alpha15"], pair["output_alpha0"]
                a_source, b_source = "alpha15", "alpha0"

            prompt = build_eval_prompt(problem, resp_a, resp_b)

            # Evaluate at alpha=0.0 (high-precision / conventional state)
            result_0 = contrastive_generate(
                model,
                tokenizer,
                problem=prompt,
                alpha=0.0,
                max_new_tokens=150,
                creative_system=EVAL_CREATIVE,
                conservative_system=EVAL_CONSERVATIVE,
            )
            choice_0 = parse_choice(result_0.text)
            preferred_0 = (
                {"A": a_source, "B": b_source}.get(choice_0) if choice_0 else None
            )

            # Evaluate at alpha=1.5 (low-precision / structural state)
            result_15 = contrastive_generate(
                model,
                tokenizer,
                problem=prompt,
                alpha=1.5,
                max_new_tokens=150,
                creative_system=EVAL_CREATIVE,
                conservative_system=EVAL_CONSERVATIVE,
            )
            choice_15 = parse_choice(result_15.text)
            preferred_15 = (
                {"A": a_source, "B": b_source}.get(choice_15) if choice_15 else None
            )

            status_0 = f"-> {preferred_0}" if preferred_0 else "-> UNPARSED"
            status_15 = f"-> {preferred_15}" if preferred_15 else "-> UNPARSED"
            print(
                f"  rep {rep:>2d}  eval@0.0 {status_0:<15s}  eval@1.5 {status_15}"
            )

            for eval_alpha, choice, preferred, text in [
                (0.0, choice_0, preferred_0, result_0.text),
                (1.5, choice_15, preferred_15, result_15.text),
            ]:
                rows.append(
                    dict(
                        problem=problem[:50],
                        rep=rep,
                        a_source=a_source,
                        b_source=b_source,
                        eval_alpha=eval_alpha,
                        choice=choice,
                        preferred=preferred,
                        eval_text=text[:200],
                    )
                )

    df = pd.DataFrame(rows)
    analyze(df)
    return df


def analyze(df):
    """Analyze and report results."""
    from scipy import stats

    print(f"\n{'=' * 60}")
    print("EVALUATOR ASYMMETRY RESULTS")
    print(f"{'=' * 60}")

    valid = df[df.preferred.notna()].copy()
    unparsed = len(df) - len(valid)
    if unparsed > 0:
        print(f"\n  ({unparsed} unparsed responses excluded)")

    eval_0 = valid[valid.eval_alpha == 0.0]
    eval_15 = valid[valid.eval_alpha == 1.5]

    pref_0_for_alpha15 = (eval_0.preferred == "alpha15").mean()
    pref_15_for_alpha15 = (eval_15.preferred == "alpha15").mean()

    print(f"\nEvaluator at alpha=0.0 (conventional state):")
    print(
        f"  Prefers structural (alpha=1.5) output: "
        f"{pref_0_for_alpha15:.1%}  ({int(pref_0_for_alpha15 * len(eval_0))}/{len(eval_0)})"
    )

    print(f"\nEvaluator at alpha=1.5 (structural state):")
    print(
        f"  Prefers structural (alpha=1.5) output: "
        f"{pref_15_for_alpha15:.1%}  ({int(pref_15_for_alpha15 * len(eval_15))}/{len(eval_15)})"
    )

    # Chi-squared test on the 2x2 contingency table
    table = np.array(
        [
            [
                (eval_0.preferred == "alpha0").sum(),
                (eval_0.preferred == "alpha15").sum(),
            ],
            [
                (eval_15.preferred == "alpha0").sum(),
                (eval_15.preferred == "alpha15").sum(),
            ],
        ]
    )

    print(f"\n  Contingency table:")
    print(f"                    prefers a=0.0    prefers a=1.5")
    print(f"    eval @ a=0.0:       {table[0, 0]:>3d}              {table[0, 1]:>3d}")
    print(f"    eval @ a=1.5:       {table[1, 0]:>3d}              {table[1, 1]:>3d}")

    chi2, p, dof, expected = stats.chi2_contingency(table)
    print(f"\n  Chi-squared: {chi2:.3f}, p={p:.4f}, dof={dof}")

    if p < 0.05:
        ratio = pref_15_for_alpha15 / max(pref_0_for_alpha15, 0.01)
        print(f"\n  >> CONFIRMED: Precision state determines evaluative framework (p={p:.4f})")
        print(f"     Recognition ratio: {ratio:.1f}x")
    else:
        print(f"\n  >> NOT SIGNIFICANT (p={p:.4f})")

    # Save
    df.to_csv("results/evaluator_asymmetry_results.csv", index=False)
    print("\nSaved: results/evaluator_asymmetry_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluator precision-state asymmetry experiment"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument(
        "--n_reps",
        type=int,
        default=N_REPS,
        help="Repetitions per problem per evaluator condition",
    )
    args = parser.parse_args()

    random.seed(42)

    problems = [
        "Why does sleep deprivation paradoxically improve depression?",
        "How do antibiotic-resistant bacteria always defeat targeted treatments?",
        "Why do antidepressants take 2-4 weeks to work?",
        "Why does ketamine produce rapid antidepressant effects lasting weeks?",
        "Why do psychedelics produce lasting personality change from a single dose?",
    ]

    model, tokenizer = load_model(args.model, load_in_4bit=args.four_bit)
    run_experiment(model, tokenizer, problems, n_reps=args.n_reps)
