from dataclasses import dataclass, field
from typing import Optional, List

from .providers.base import BaseLLMProvider, GenerationConfig
from .evaluator.prompts import CREATIVE_SYSTEM, FREE_CREATIVE_SYSTEM, CONSERVATIVE_SYSTEM, DELTA_SYSTEM
from .pss import PSS, PSSResult


@dataclass
class PRISMResult:
    synthesis: str
    pss: PSSResult
    domain_a: str
    domain_b: str
    problem: str
    attempts: int = 1


@dataclass
class PRISMConfig:
    min_pss: float = 0.04
    max_attempts: int = 6
    creative_temperature: float = 1.0
    conservative_temperature: float = 0.2
    max_tokens: int = 1500
    verbose: bool = False


class PRISM:
    """
    PRISM — Precision Reduction for Isomorphic Synthesis and Mapping

    Finds structural equivalences between distant scientific domains
    using contrastive precision reduction.

    Usage:
        from prism import PRISM
        from prism.providers import AnthropicProvider

        engine = PRISM(
            generator=AnthropicProvider(),
            evaluator=AnthropicProvider()
        )

        result = engine.synthesize(
            problem="Why does sleep deprivation paradoxically improve depression?",
            domain_a="neuroscience",
            domain_b="information theory"
        )

        print(f"PSS: {result.pss.pss:.3f}")
        print(result.synthesis)
    """

    def __init__(
        self,
        generator: BaseLLMProvider,
        evaluator: Optional[BaseLLMProvider] = None,
        config: Optional[PRISMConfig] = None,
    ):
        self.generator = generator
        self.evaluator = evaluator or generator
        self.config = config or PRISMConfig()
        self.pss_scorer = PSS(generator=generator, evaluator=self.evaluator)

    def _creative_pass(self, problem: str, domain_a: str, domain_b: str) -> str:
        prompt = (
            f"Find the deep structural connection between {domain_a} and {domain_b} "
            f"that illuminates this problem:\n\n{problem}\n\n"
            f"What formal framework, mathematical structure, or organizing principle "
            f"from {domain_b} maps onto the problem in {domain_a}? "
            f"What does this transfer reveal that wasn't visible before?"
        )

        config = GenerationConfig(
            temperature=self.config.creative_temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=CREATIVE_SYSTEM,
        )
        return self.generator.generate(prompt, config).content

    def _free_creative_pass(self, problem: str) -> str:
        prompt = (
            f"What hidden formal structure, mathematical framework, or "
            f"organizing principle explains this problem:\n\n{problem}\n\n"
            f"Look for deep structural patterns — equations, optimization "
            f"objectives, constraint structures — that reveal something "
            f"non-obvious. What does the structure tell you that standard "
            f"analysis misses?"
        )

        config = GenerationConfig(
            temperature=self.config.creative_temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=FREE_CREATIVE_SYSTEM,
        )
        return self.generator.generate(prompt, config).content

    def _conservative_pass(self, problem: str, domain_a: str) -> str:
        prompt = f"What is the standard scientific understanding of: {problem}"

        config = GenerationConfig(
            temperature=self.config.conservative_temperature,
            max_tokens=800,
            system_prompt=CONSERVATIVE_SYSTEM,
        )
        return self.generator.generate(prompt, config).content

    def _delta_extraction(self, creative: str, conservative: str) -> str:
        prompt = (
            f"CREATIVE RESPONSE:\n{creative}\n\n"
            f"CONVENTIONAL RESPONSE:\n{conservative}\n\n"
            f"Extract only the ideas present in the creative response "
            f"but absent from the conventional response."
        )

        config = GenerationConfig(
            temperature=0.3,
            max_tokens=800,
            system_prompt=DELTA_SYSTEM,
        )
        return self.generator.generate(prompt, config).content

    def synthesize(
        self,
        problem: str,
        domain_a: str,
        domain_b: str,
    ) -> PRISMResult:
        """
        Find a high-PSS structural synthesis for the given problem.

        Args:
            problem: The scientific or technical problem to solve
            domain_a: The source domain (where the problem lives)
            domain_b: The target domain (where the solution might come from)

        Returns:
            PRISMResult with synthesis and PSS score
        """
        best_result = None

        for attempt in range(1, self.config.max_attempts + 1):
            if self.config.verbose:
                print(f"Attempt {attempt}/{self.config.max_attempts}...")

            # Three-stage contrastive mechanism
            creative = self._creative_pass(problem, domain_a, domain_b)
            conservative = self._conservative_pass(problem, domain_a)
            synthesis = self._delta_extraction(creative, conservative)

            # Score with PSS
            pss_result = self.pss_scorer.score(
                response=synthesis,
                domain_a=domain_a,
                domain_b=domain_b,
            )

            if self.config.verbose:
                print(
                    f"  PSS={pss_result.pss:.4f} "
                    f"(AD={pss_result.attractor_distance:.4f}, "
                    f"coh={pss_result.coherence:.4f})"
                )

            if best_result is None or pss_result.pss > best_result.pss.pss:
                best_result = PRISMResult(
                    synthesis=synthesis,
                    pss=pss_result,
                    domain_a=domain_a,
                    domain_b=domain_b,
                    problem=problem,
                    attempts=attempt,
                )

            # Early stop if quality gate met
            if pss_result.pss >= self.config.min_pss:
                break

        return best_result

    def free_synthesis(
        self,
        problem: str,
        domain: str,
    ) -> PRISMResult:
        """
        Run contrastive synthesis without domain specification.

        The creative pass asks for novel structural explanations generally,
        without mentioning specific domains. The conservative pass and delta
        extraction work as usual. PSS scoring uses the given domain as both
        domain_a and domain_b.

        Args:
            problem: The scientific or technical problem to solve
            domain: The domain the problem lives in (used for conservative
                    pass and PSS scoring)

        Returns:
            PRISMResult with synthesis and PSS score
        """
        best_result = None

        for attempt in range(1, self.config.max_attempts + 1):
            if self.config.verbose:
                print(f"Attempt {attempt}/{self.config.max_attempts}...")

            # Three-stage contrastive mechanism (domain-free creative)
            creative = self._free_creative_pass(problem)
            conservative = self._conservative_pass(problem, domain)
            synthesis = self._delta_extraction(creative, conservative)

            # Score with PSS — use domain as both a and b
            pss_result = self.pss_scorer.score(
                response=synthesis,
                domain_a=domain,
                domain_b=domain,
            )

            if self.config.verbose:
                print(
                    f"  PSS={pss_result.pss:.4f} "
                    f"(AD={pss_result.attractor_distance:.4f}, "
                    f"coh={pss_result.coherence:.4f})"
                )

            if best_result is None or pss_result.pss > best_result.pss.pss:
                best_result = PRISMResult(
                    synthesis=synthesis,
                    pss=pss_result,
                    domain_a=domain,
                    domain_b=domain,
                    problem=problem,
                    attempts=attempt,
                )

            # Early stop if quality gate met
            if pss_result.pss >= self.config.min_pss:
                break

        return best_result

    def search(
        self,
        problem: str,
        domain_a: str,
        target_domains: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[PRISMResult]:
        """
        Search across multiple target domains and return top-k results by PSS.

        Args:
            problem: The problem to solve
            domain_a: The source domain
            target_domains: List of domains to search (uses default list if None)
            top_k: Number of top results to return

        Returns:
            List of PRISMResult sorted by PSS descending
        """
        from .domains import DEFAULT_DOMAINS

        domains = target_domains or DEFAULT_DOMAINS
        results = []

        for domain_b in domains:
            if domain_b.lower() == domain_a.lower():
                continue
            if self.config.verbose:
                print(f"\nSearching: {domain_a} x {domain_b}")
            try:
                result = self.synthesize(problem, domain_a, domain_b)
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    print(f"  Failed: {e}")

        results.sort(key=lambda r: r.pss.pss, reverse=True)
        return results[:top_k]
