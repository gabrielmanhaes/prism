"""
Cognitive mode definitions for PRISM.

Each mode specifies a creative/conservative prompt pair and the empirically
confirmed optimal alpha for that task class. The alpha reflects the depth
of the conventional attractor that must be escaped for each cognitive task.
"""

from dataclasses import dataclass


@dataclass
class CognitiveMode:
    """
    A cognitive mode defines the prompt pair and optimal alpha
    for a specific class of cognitive task.

    The creative/conservative prompt pair determines which dimension
    of the model's representational space gets amplified.
    The optimal_alpha reflects the depth of the conventional attractor
    for this task class.
    """

    name: str
    description: str
    creative_system: str
    conservative_system: str
    optimal_alpha: float
    alpha_range: tuple
    task_class: str  # 'synthesis' | 'forensic' | 'analytical' | 'security'


MODES = {
    "synthesis": CognitiveMode(
        name="Scientific synthesis",
        description=(
            "Cross-domain structural transfer. Finds mathematical "
            "equivalences and formal isomorphisms between distant scientific "
            "domains. Optimal for hypothesis generation and novel mechanism "
            "discovery."
        ),
        creative_system=(
            "You are a scientific synthesis engine. Find the deep structural "
            "connection between these two domains. Look for mathematical "
            "equivalences, formal isomorphisms, and organizing principles "
            "that transfer entire problem-solving frameworks between fields. "
            "Be mechanistically specific — name the formal tools that transfer."
        ),
        conservative_system=(
            "Provide a conventional, standard, textbook response. Focus on "
            "well-established facts and mainstream interpretations only. "
            "Do not speculate or introduce novel ideas."
        ),
        optimal_alpha=1.5,
        alpha_range=(1.0, 2.0),
        task_class="synthesis",
    ),
    "forensic": CognitiveMode(
        name="Forensic pattern detection",
        description=(
            "Anomaly detection and non-obvious pattern recognition. "
            "Surfaces what the conventional coherence-preserving analysis "
            "filters. Optimal for detecting anomalies in data, arguments, "
            "or code."
        ),
        creative_system=(
            "You are searching for hidden patterns, anomalies, and "
            "non-obvious connections. What is present that a conventional "
            "analysis would miss? Look for what doesn't fit, what's "
            "inconsistent, what the surface reading suppresses."
        ),
        conservative_system=(
            "Perform a standard methodical analysis. State only what is "
            "directly supported by the evidence. Provide the conventional "
            "interpretation."
        ),
        optimal_alpha=0.5,
        alpha_range=(0.25, 0.75),
        task_class="forensic",
    ),
    "uncertainty": CognitiveMode(
        name="Calibrated uncertainty",
        description=(
            "Surfaces genuine uncertainty that the high-P confident "
            "conventional response suppresses. Identifies where scientific "
            "or empirical consensus is actually contested or unknown."
        ),
        creative_system=(
            "Identify exactly what is not known about this topic. Where are "
            "the genuine gaps, the contested claims, the places where "
            "confident answers are not yet warranted? Be specific about the "
            "uncertainty structure — name the competing positions and why "
            "they remain unresolved."
        ),
        conservative_system=(
            "Provide a confident, comprehensive answer to this question "
            "based on current knowledge. Summarize what experts agree on."
        ),
        optimal_alpha=0.5,
        alpha_range=(0.25, 0.75),
        task_class="analytical",
    ),
    "contradiction": CognitiveMode(
        name="Contradiction detection",
        description=(
            "Breaks the coherence-preserving reading mode to surface "
            "logical tensions between statements. Optimal for finding "
            "inconsistencies in arguments, documents, or codebases."
        ),
        creative_system=(
            "Find internal tensions, inconsistencies, and statements that "
            "cannot simultaneously be true. Focus on logical "
            "incompatibilities — where does this text contradict itself, "
            "even indirectly through implication?"
        ),
        conservative_system=(
            "Summarize the main claims of this text. Assume it is "
            "internally consistent and focus on understanding its overall "
            "meaning."
        ),
        optimal_alpha=1.0,
        alpha_range=(0.75, 1.25),
        task_class="analytical",
    ),
    "adversarial": CognitiveMode(
        name="Adversarial analysis",
        description=(
            "Suppresses the cooperative charitable-reading default to "
            "surface genuine structural weaknesses. RLHF trains heavy "
            "cooperation — this mode requires stronger contrast to break "
            "that attractor."
        ),
        creative_system=(
            "Find the specific assumption that, if false, destroys this "
            "argument. What is the load-bearing premise that everyone "
            "accepts but has not been tested? Where is the structural "
            "weakness?"
        ),
        conservative_system=(
            "Summarize this argument charitably, identifying its strongest "
            "points and most solid empirical support. What makes it "
            "compelling?"
        ),
        optimal_alpha=1.0,
        alpha_range=(0.75, 1.5),
        task_class="analytical",
    ),
    "depth": CognitiveMode(
        name="Mechanistic depth",
        description=(
            "Suppresses the comprehensive-overview attractor to force "
            "commitment to one central insight developed deeply. The "
            "breadth-over-depth default is one of the most strongly "
            "RLHF-trained behaviors."
        ),
        creative_system=(
            "Identify the single most important mechanism that explains "
            "this phenomenon — the one insight that, if understood deeply, "
            "makes everything else fall into place. Go deep on that "
            "mechanism only. Name the specific causal chain."
        ),
        conservative_system=(
            "Provide a comprehensive overview of this topic covering all "
            "the main aspects, mechanisms, and considerations."
        ),
        optimal_alpha=1.5,
        alpha_range=(1.0, 2.0),
        task_class="analytical",
    ),
    "temporal": CognitiveMode(
        name="Temporal reasoning",
        description=(
            "Suppresses the immediate/proximate cause attractor to surface "
            "second and third-order systemic consequences."
        ),
        creative_system=(
            "Consider the long-term systemic implications and second-order "
            "effects that play out over years or decades. What does this "
            "look like from a 10-year perspective? What downstream "
            "consequences emerge that proximate analysis misses?"
        ),
        conservative_system=(
            "Focus on the immediate, direct, first-order effects. What "
            "happens in the short term? What are the proximate causes and "
            "effects?"
        ),
        optimal_alpha=0.5,
        alpha_range=(0.25, 0.75),
        task_class="analytical",
    ),
    "analogy": CognitiveMode(
        name="Deep analogical construction",
        description=(
            "Maximum structural stretch — finds precise formal mappings "
            "between the most distant domains. Requires the strongest "
            "conventional suppression because every token fights the "
            "model's distributional prior to stay within one domain."
        ),
        creative_system=(
            "Find the structural pattern that recurs across completely "
            "different domains. What is the formal isomorphism — the "
            "mapping where solving the problem in domain B gives you the "
            "solution in domain A? Be precise about what maps to what."
        ),
        conservative_system=(
            "Describe this phenomenon at its most natural level of "
            "analysis. What is the standard unit of description for this "
            "field?"
        ),
        optimal_alpha=2.0,
        alpha_range=(1.5, 2.0),
        task_class="synthesis",
    ),
    "security_injection": CognitiveMode(
        name="Security — injection and crypto",
        description=(
            "Targets SQL injection, command injection, cryptographic "
            "weaknesses, and predictable token generation."
        ),
        creative_system=(
            "You are a security researcher. Assume this code contains "
            "exploitable vulnerabilities. Find SQL injection via string "
            "concatenation or interpolation, command injection via system "
            "calls with user input, weak hash functions, predictable "
            "tokens or salts, and hardcoded secrets. Name the exact "
            "function, input, and outcome an attacker achieves."
        ),
        conservative_system=(
            "You are a developer explaining this code to a new team "
            "member. Explain how user input is safely handled, how "
            "database queries are parameterized, and how tokens are "
            "securely generated."
        ),
        optimal_alpha=1.0,
        alpha_range=(0.75, 1.25),
        task_class="security",
    ),
    "security_authorization": CognitiveMode(
        name="Security — authorization and logic",
        description=(
            "Targets IDOR, broken access control, missing permission "
            "checks, and business logic errors."
        ),
        creative_system=(
            "Find authorization flaws, IDOR, and broken access control. "
            "Look for missing permission checks, places where callers can "
            "access data they shouldn't, assumptions about the caller that "
            "an attacker can violate, and operations that should be "
            "restricted but aren't."
        ),
        conservative_system=(
            "Explain the authorization model. Who is allowed to access "
            "what resources, and how does the code enforce those "
            "boundaries?"
        ),
        optimal_alpha=1.0,
        alpha_range=(0.75, 1.25),
        task_class="security",
    ),
    "security_threading": CognitiveMode(
        name="Security — race conditions and threading",
        description=(
            "Targets TOCTOU vulnerabilities, thread-safety violations, "
            "and non-atomic operations on shared state."
        ),
        creative_system=(
            "Find race conditions and thread-safety violations. Consider "
            "concurrent execution: what happens when two threads execute "
            "simultaneously? Look for shared state accessed without locks, "
            "check-then-act patterns, and operations that appear atomic "
            "but aren't."
        ),
        conservative_system=(
            "Explain the synchronization design and why shared state is "
            "safe for concurrent access. Where are locks used and what do "
            "they protect?"
        ),
        optimal_alpha=1.0,
        alpha_range=(0.75, 1.25),
        task_class="security",
    ),
}


def get_mode(name: str) -> CognitiveMode:
    """Get a cognitive mode by name."""
    if name not in MODES:
        available = ", ".join(sorted(MODES.keys()))
        raise ValueError(f"Unknown mode '{name}'. Available: {available}")
    return MODES[name]


def list_modes() -> dict:
    """Return summary of all available modes."""
    return {
        name: {
            "description": mode.description.strip(),
            "optimal_alpha": mode.optimal_alpha,
            "task_class": mode.task_class,
        }
        for name, mode in MODES.items()
    }
