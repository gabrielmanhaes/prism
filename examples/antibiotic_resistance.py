"""
PRISM — Antibiotic Resistance Case Study
==========================================
Search for structural explanations of why targeted treatments fail.
"""

from prism import PRISM, PRISMConfig


def main():
    engine = PRISM.from_env(config=PRISMConfig(verbose=True, max_attempts=3))

    results = engine.search(
        problem="How do antibiotic-resistant bacteria always defeat targeted treatments?",
        domain_a="microbiology",
        target_domains=[
            "cybernetics",
            "evolutionary biology",
            "game theory",
            "information theory",
            "immunology",
        ],
        top_k=3,
    )

    print(f"\nTop {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"#{i} {r.domain_a} x {r.domain_b}")
        print(f"PSS={r.pss.pss:.4f}")
        print(r.synthesis[:500] + "..." if len(r.synthesis) > 500 else r.synthesis)


if __name__ == "__main__":
    main()
