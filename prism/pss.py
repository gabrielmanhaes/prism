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
    # v0.2 fields
    raw_attractor_distance: float = 0.0
    domain_pair_distance: float = 0.0
    has_formal_tool_transfer: bool = False
    is_novel_bridge: bool = False


VERDICT_SCORES = {
    "STRUCTURAL_ISOMORPHISM": 1.0,
    "STRUCTURAL_REVELATION": 0.8,
    "GENUINE_INSIGHT": 0.6,
    "INTERESTING_FRAMING": 0.4,
    "SUPERFICIAL": 0.2,
    "NOTHING_NEW": 0.1,
}

TOOL_TRANSFER_BOOST = 1.25
ESTABLISHED_PENALTY = 0.5


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

    Attractor Distance: how far the response departs from conventional responses,
        normalized by the semantic distance between the two domains.
    Structural Coherence: whether the departure reveals genuine structure in both
        domains, with tiebreakers for formal tool transfer and literature novelty.
    """

    def __init__(
        self,
        generator: BaseLLMProvider,
        evaluator: BaseLLMProvider,
        n_baseline: int = 8,
        baseline_temperature: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_coherence_evals: int = 2,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.n_baseline = n_baseline
        self.baseline_temperature = baseline_temperature
        self._embedder = None
        self._embedding_model = embedding_model
        self.n_coherence_evals = n_coherence_evals

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

    def _domain_pair_distance(self, domain_a: str, domain_b: str) -> float:
        """Compute semantic distance between two domain names."""
        if domain_a.lower() == domain_b.lower():
            return 0.0
        embedder = self._get_embedder()
        embeddings = embedder.encode([domain_a, domain_b])
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        return float(np.clip((1 - cos_sim) / 2.0, 0, 1))

    def _structural_coherence(
        self, response: str, domain_a: str, domain_b: str
    ) -> tuple:
        """
        Evaluate structural coherence using five-question categorical protocol.
        Returns (coherence_score, verdict_a, verdict_b, has_prediction,
                 has_tool_transfer, is_novel_bridge)
        """
        cleaned = self._strip_meta(response)

        eval_prompt = f"""Evaluate this cross-domain connection between {domain_a} and {domain_b}.

CONNECTION:
{cleaned}

Answer these five questions:

Q1: Does this connection generate a non-obvious testable prediction?
Answer: YES or NO

Q2: Would an expert in {domain_a} learn something genuinely new about {domain_a} from this connection?
Choose exactly one: STRUCTURAL_ISOMORPHISM | STRUCTURAL_REVELATION | GENUINE_INSIGHT | INTERESTING_FRAMING | SUPERFICIAL | NOTHING_NEW

Q3: Would an expert in {domain_b} learn something genuinely new about {domain_b} from this connection?
Choose exactly one: STRUCTURAL_ISOMORPHISM | STRUCTURAL_REVELATION | GENUINE_INSIGHT | INTERESTING_FRAMING | SUPERFICIAL | NOTHING_NEW

Q4: Does this connection transfer a specific formal tool (equation, algorithm, measurement protocol, or theorem) from {domain_b} that can be directly applied in {domain_a}?
Answer: YES or NO

Q5: Is this connection already well-established in the literature connecting these two fields, or does it represent a genuinely novel bridge?
Answer: ESTABLISHED or NOVEL

Format your response as:
Q1: [YES/NO]
Q2: [VERDICT]
Q3: [VERDICT]
Q4: [YES/NO]
Q5: [ESTABLISHED/NOVEL]"""

        config = GenerationConfig(temperature=0.1, max_tokens=200)

        try:
            result = self.evaluator.generate(eval_prompt, config)
            return self._parse_coherence(result.content)
        except Exception:
            return 0.02, "SUPERFICIAL", "SUPERFICIAL", False, False, True

    def _parse_coherence(self, response: str) -> tuple:
        """Parse coherence evaluation response into score and components."""
        lines = response.strip().split("\n")

        has_prediction = False
        verdict_a = "SUPERFICIAL"
        verdict_b = "SUPERFICIAL"
        has_tool_transfer = False
        is_novel_bridge = True  # default to novel (no penalty) if unparseable

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
            elif line.startswith("Q4:"):
                has_tool_transfer = "YES" in line.upper()
            elif line.startswith("Q5:"):
                is_novel_bridge = "NOVEL" in line.upper() and "ESTABLISHED" not in line.upper()

        score = VERDICT_SCORES[verdict_a] * VERDICT_SCORES[verdict_b]
        if has_tool_transfer:
            score *= TOOL_TRANSFER_BOOST
        if not is_novel_bridge:
            score *= ESTABLISHED_PENALTY
        score = float(np.clip(score, 0, 1))

        return score, verdict_a, verdict_b, has_prediction, has_tool_transfer, is_novel_bridge

    def score(
        self, response: str, domain_a: str, domain_b: str
    ) -> PSSResult:
        """Compute PSS for a single response."""
        # Attractor distance
        raw_ad = self._attractor_distance(response, domain_a, domain_b)

        # Normalize AD by domain pair distance (#6)
        dpd = self._domain_pair_distance(domain_a, domain_b)
        if dpd > 0.01:
            ad = float(np.clip(raw_ad / dpd, 0, 1))
        else:
            ad = raw_ad

        # Structural coherence — run N times and average (#2)
        runs = []
        for _ in range(self.n_coherence_evals):
            runs.append(self._structural_coherence(response, domain_a, domain_b))

        coherence = float(np.mean([r[0] for r in runs]))
        best_run = max(runs, key=lambda r: r[0])
        verdict_a = best_run[1]
        verdict_b = best_run[2]
        has_prediction = any(r[3] for r in runs)
        has_tool_transfer = any(r[4] for r in runs)
        is_novel_bridge = any(r[5] for r in runs)

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
            raw_attractor_distance=raw_ad,
            domain_pair_distance=dpd,
            has_formal_tool_transfer=has_tool_transfer,
            is_novel_bridge=is_novel_bridge,
        )
