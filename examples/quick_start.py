"""
PRISM Quick Start
================
Find structural equivalences between scientific domains.
"""

from prism import PRISM, PRISMConfig


def main():
    # Auto-detect provider from environment
    # Needs ANTHROPIC_API_KEY or OPENAI_API_KEY in environment
    engine = PRISM.from_env(config=PRISMConfig(verbose=True, max_attempts=3))

    # Single domain pair synthesis
    print("=" * 60)
    print("PRISM — Single Domain Synthesis")
    print("=" * 60)

    result = engine.synthesize(
        problem="Why does sleep deprivation paradoxically improve depression?",
        domain_a="neuroscience",
        domain_b="information theory",
    )

    print(f"\nPSS: {result.pss.pss:.4f}")
    print(f"Attractor Distance: {result.pss.attractor_distance:.4f}")
    print(f"Coherence: {result.pss.coherence:.4f} ({result.pss.coherence_verdict})")
    print(f"Testable prediction: {result.pss.has_testable_prediction}")
    print(f"\nSynthesis:\n{result.synthesis}")


if __name__ == "__main__":
    main()
