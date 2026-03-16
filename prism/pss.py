import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .providers.base import BaseLLMProvider, GenerationConfig


@dataclass
class PSSResult:
    pss: float
    attractor_distance: float
    coherence: float
    coherence_verdict: str
    domain_a_verdict: str
    domain_b_verdict: str
    has_testable_prediction: bool
    response: str


VERDICT_SCORES = {
    "STRUCTURAL_ISOMORPHISM": 1.0,
    "STRUCTURAL_REVELATION": 0.8,
    "GENUINE_INSIGHT": 0.6,
    "INTERESTING_FRAMING": 0.4,
    "SUPERFICIAL": 0.2,
    "NOTHING_NEW": 0.1,
}


# Meta-commentary patterns that inflate distance without adding content
_META_PREFIXES = [
    r"^The novel (?:element|insight|idea|connection|concept)\b.*?(?:is that |lies in |is: |absent from the conventional response is:? ?|missing from the conventional response is:? ?)",
    r"^The creative response (?:introduces the idea of |uniquely emphasizes |suggests |highlights )",
    r"^The unique insight\b.*?(?:is that |is: |from the creative response is:? ?)",
    r"^(?:Here is |)(?:The |A )(?:genuinely |)novel (?:insight|element)\b.*?(?:is that |is: )",
    r'^"?The creative (?:insight|response)\b[^"]*?(?:is:? ?|absent (?:from|in) the conventional response is:? ?)',
]

_META_SUFFIXES = [
    r"\s*This (?:is not|additional|novel|unique)[\w\s]*(?:explicitly stated|layer|perspective)[\w\s.]*$",
    r"\s*Thus,? the novel insight[\w\s:]*$",
]


class PSS:
    """
    Precision-Survival Score — measures cross-domain structural synthesis quality.

    PSS = Attractor Distance x Structural Coherence

    Attractor Distance: how far the response departs from conventional responses
    Structural Coherence: whether the departure reveals genuine structure in both domains
    """

    def __init__(
        self,
        generator: BaseLLMProvider,
        evaluator: BaseLLMProvider,
        n_baseline: int = 8,
        baseline_temperature: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.n_baseline = n_baseline
        self.baseline_temperature = baseline_temperature
        self._embedder = None
        self._embedding_model = embedding_model

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def _strip_meta(self, text: str) -> str:
        """Remove meta-commentary that inflates distance without adding content."""
        stripped = text.strip()

        # Remove wrapping quotes
        if stripped.startswith('"') and stripped.endswith('"'):
            stripped = stripped[1:-1].strip()

        for pattern in _META_PREFIXES:
            stripped = re.sub(pattern, "", stripped, count=1, flags=re.IGNORECASE)

        for pattern in _META_SUFFIXES:
            stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)

        stripped = stripped.strip().lstrip(":").lstrip('"').strip()

        # If stripping removed everything, return original
        if len(stripped) < 20:
            return text.strip()

        return stripped

    def _attractor_distance(
        self, response: str, domain_a: str, domain_b: str
    ) -> float:
        """Compute semantic distance from conventional response centroid."""
        embedder = self._get_embedder()

        baseline_prompts = [
            f"Name one thing that {domain_a} and {domain_b} have in common.",
            f"What is a standard connection between {domain_a} and {domain_b}?",
            f"Give a typical example of how {domain_a} relates to {domain_b}.",
            f"What do experts in {domain_a} usually say about {domain_b}?",
            f"Describe the most obvious link between {domain_a} and {domain_b}.",
            f"What is the conventional wisdom connecting {domain_a} and {domain_b}?",
            f"Give a textbook example of {domain_a} applied to {domain_b}.",
            f"What is the most common analogy between {domain_a} and {domain_b}?",
        ]

        baseline_config = GenerationConfig(
            temperature=self.baseline_temperature,
            max_tokens=250,
            system_prompt="Provide a conventional, textbook-style response.",
        )

        baselines = []
        for bp in baseline_prompts[: self.n_baseline]:
            try:
                result = self.generator.generate(bp, baseline_config)
                baselines.append(result.content)
            except Exception:
                continue

        if len(baselines) < 3:
            return 0.5

        # Strip meta from response before embedding
        cleaned = self._strip_meta(response)

        all_texts = baselines + [cleaned]
        embeddings = embedder.encode(all_texts)

        centroid = embeddings[:-1].mean(axis=0)
        response_emb = embeddings[-1]

        cos_sim = np.dot(centroid, response_emb) / (
            np.linalg.norm(centroid) * np.linalg.norm(response_emb) + 1e-8
        )
        return float(np.clip((1 - cos_sim) / 2.0, 0, 1))

    def _structural_coherence(
        self, response: str, domain_a: str, domain_b: str
    ) -> tuple:
        """
        Evaluate structural coherence using three-question categorical protocol.
        Returns (coherence_score, verdict_a, verdict_b, has_prediction)
        """
        cleaned = self._strip_meta(response)

        eval_prompt = f"""Evaluate this cross-domain connection between {domain_a} and {domain_b}.

CONNECTION:
{cleaned}

Answer these three questions:

Q1: Does this connection generate a non-obvious testable prediction?
Answer: YES or NO

Q2: Would an expert in {domain_a} learn something genuinely new about {domain_a} from this connection?
Choose exactly one: STRUCTURAL_ISOMORPHISM | STRUCTURAL_REVELATION | GENUINE_INSIGHT | INTERESTING_FRAMING | SUPERFICIAL | NOTHING_NEW

Q3: Would an expert in {domain_b} learn something genuinely new about {domain_b} from this connection?
Choose exactly one: STRUCTURAL_ISOMORPHISM | STRUCTURAL_REVELATION | GENUINE_INSIGHT | INTERESTING_FRAMING | SUPERFICIAL | NOTHING_NEW

Format your response as:
Q1: [YES/NO]
Q2: [VERDICT]
Q3: [VERDICT]"""

        config = GenerationConfig(temperature=0.1, max_tokens=200)

        try:
            result = self.evaluator.generate(eval_prompt, config)
            return self._parse_coherence(result.content)
        except Exception:
            return 0.02, "SUPERFICIAL", "SUPERFICIAL", False

    def _parse_coherence(self, response: str) -> tuple:
        """Parse coherence evaluation response."""
        lines = response.strip().split("\n")

        has_prediction = False
        verdict_a = "SUPERFICIAL"
        verdict_b = "SUPERFICIAL"

        for line in lines:
            line = line.strip()
            if line.startswith("Q1:"):
                has_prediction = "YES" in line.upper()
            elif line.startswith("Q2:"):
                for v in VERDICT_SCORES:
                    if v in line:
                        verdict_a = v
                        break
            elif line.startswith("Q3:"):
                for v in VERDICT_SCORES:
                    if v in line:
                        verdict_b = v
                        break

        score = VERDICT_SCORES[verdict_a] * VERDICT_SCORES[verdict_b]
        return score, verdict_a, verdict_b, has_prediction

    def score(
        self, response: str, domain_a: str, domain_b: str
    ) -> PSSResult:
        """Compute PSS for a single response."""
        ad = self._attractor_distance(response, domain_a, domain_b)
        coherence, verdict_a, verdict_b, has_prediction = (
            self._structural_coherence(response, domain_a, domain_b)
        )
        pss = ad * coherence

        return PSSResult(
            pss=pss,
            attractor_distance=ad,
            coherence=coherence,
            coherence_verdict=f"{verdict_a} x {verdict_b}",
            domain_a_verdict=verdict_a,
            domain_b_verdict=verdict_b,
            has_testable_prediction=has_prediction,
            response=response,
        )
