"""
PRISM — Psychiatry Case Study
==============================
Explore sleep deprivation and depression through multiple domain lenses.
"""

from prism import PRISM, PRISMConfig


def main():
    engine = PRISM.from_env(config=PRISMConfig(verbose=True, max_attempts=3))

    results = engine.search(
        problem="Why does sleep deprivation paradoxically improve depression?",
        domain_a="neuroscience",
        target_domains=[
            "information theory",
            "statistical mechanics",
            "control theory",
            "evolutionary biology",
            "dynamical systems",
        ],
        top_k=3,
    )

    print(f"\nTop {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"#{i} {r.domain_a} x {r.domain_b}")
        print(f"PSS={r.pss.pss:.4f} (AD={r.pss.attractor_distance:.4f}, coh={r.pss.coherence:.4f})")
        print(f"Verdict: {r.pss.coherence_verdict}")
        print(f"Testable: {r.pss.has_testable_prediction}")
        print(f"\n{r.synthesis}")


if __name__ == "__main__":
    main()
