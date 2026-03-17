# PRISM

**Precision Reduction for Isomorphic Synthesis and Mapping**

Token-level contrastive generation that modulates the cognitive state
of language models — not just what they generate, but how they evaluate.

---

## What this is

Language models have internal cognitive states. The same model, with the
same knowledge, produces systematically different outputs and makes
systematically different evaluative judgments depending on its precision
state. PRISM is a mechanism for accessing and controlling those states.

The core operation: at every token generation step, run two forward passes
through the same model — one with a creative system prompt, one with a
conservative system prompt. Combine the resulting probability distributions
contrastively:

```
score(token) = log P_creative(token) - alpha * log P_conservative(token)
```

The `alpha` parameter controls how aggressively the conventional
distribution is suppressed. Higher alpha = more precision reduction =
deeper departure from conventional outputs.

This extends Li et al. 2023 (Contrastive Decoding) in a specific direction:
rather than contrasting model sizes to amplify capability, PRISM contrasts
cognitive state prompts on the same model to access different regions of
its representational space.

---

## The alpha spectrum

Different cognitive tasks have different optimal alpha values because
they require different amounts of precision reduction to escape the
conventional attractor:

```
alpha=0.0  ──────────────────────────────────────────────────  alpha=2.0
  │                                                                  │
  │  High-P                                                  Low-P  │
  │  Conventional                                         Generative │
  │                                                                  │
  0.5          1.0              1.5              2.0
   │            │                │                │
Forensics   Contradiction    Synthesis        Analogy
Uncertainty  Adversarial      Depth
Temporal
```

Empirically confirmed optimal alpha values:

| Task | Optimal alpha | Mechanism suppressed |
|------|--------------|---------------------|
| Forensic pattern detection | 0.5 | Coherence-preserving reading |
| Calibrated uncertainty | 0.5 | Confident-comprehensive default |
| Temporal reasoning | 0.5 | Proximate-cause attractor |
| Contradiction detection | 1.0 | Coherence confabulation |
| Adversarial analysis | 1.0 | Cooperative charitable reading |
| Scientific synthesis | 1.5 | Conventional explanation attractor |
| Mechanistic depth | 1.5 | Breadth-over-depth RLHF artifact |
| Analogical construction | 2.0 | Single-domain distributional prior |

---

## The PSS metric

PSS (Precision-Survival Score) measures structural synthesis quality:

```
PSS = Attractor Distance × Structural Coherence
```

**Attractor Distance (AD):** cosine distance from the centroid of
conventional baseline responses. Measures how far the output departs
from the conventional distribution. Computed using sentence embeddings.

**Structural Coherence:** estimated by an evaluator asking three questions:
1. Would a domain expert learn something non-obvious from this?
2. Does this identify a structural mechanism that transfers across domains?
3. Does it generate a testable prediction?

Outputs are classified as: NOTHING_NEW -> SURFACE_NOVELTY ->
GENUINE_INSIGHT -> STRUCTURAL_REVELATION

**PSS is calibrated for synthesis mode.** For other cognitive modes,
use task-appropriate metrics (hit rate for forensics, binary correctness
for contradiction detection, etc.).

**Known limitation:** The coherence evaluator operates at its default
high-P state, which systematically underestimates structural synthesis
quality. This connects to the evaluator asymmetry finding below.

---

## The main finding: evaluator asymmetry

The most important result is not about generation — it's about evaluation.

**Setup:** Generate outputs at alpha=0.0 (conventional) and alpha=1.5
(structural synthesis) for the same problems. Then ask the model to
evaluate which output is better — once at alpha=0.0 (high-P evaluator)
and once at alpha=1.5 (low-P evaluator). The outputs being evaluated
are identical in both conditions.

**Result:**

```
Evaluator state         Prefers α=0.0    Prefers α=1.5
──────────────────────────────────────────────────────
eval @ α=0.0 (high-P)   95%  (38/40)     5%   (2/40)
eval @ α=1.5 (low-P)    65%  (26/40)    35%  (14/40)
──────────────────────────────────────────────────────
chi-squared = 9.45, p = 0.0021
```

The model in its high-P conventional state recognizes structural
synthesis as better only 5% of the time. The same model, evaluating
the same outputs, in its low-P state recognizes structural synthesis
as better 35% of the time — a 7x increase.

**This is not a prompt effect.** Both conditions use identical system
prompts. The only variable is alpha. The result demonstrates that the
model's precision state determines its evaluative framework — not just
what it generates.

**Implication for RLHF:** Human preference raters evaluate model outputs
in a high-P conventional state. They prefer clear, confident, conventional
responses — the outputs that alpha=0.0 produces. RLHF then trains the model
to produce more of those outputs. The evaluative asymmetry result suggests
this process filters genuine structural insights at approximately a 19:1
ratio relative to a low-P evaluator.

---

## Prompt pair matters — but so does alpha

Running the mechanism with a generic system prompt at alpha=1.5 produces
**worse** results than no mechanism at all. The mechanism amplifies whatever
direction the creative prompt establishes — including its blind spots.

The creative/conservative prompt pair is privileged: it is the only
prompt pair that produces consistent PSS improvement with increasing
alpha. Seven other contrast axes (temporal, minority view, failure mode,
mechanistic depth, cross-level, adversarial) produced flat or declining
quality when scored on PSS.

---

## Ablation: contrastive vs specialized prompt alone

**Qwen 3B judge, N=40 per condition:**

```
Condition               Score (1-5)   Distribution
────────────────────────────────────────────────────
A: Baseline             3.85 ± 1.42   Mixed (1-5)
B: Specialized prompt   4.45 ± 0.55   Mostly 4-5
C: Contrastive α=1.5   4.67 ± 0.47   13×4, 27×5
────────────────────────────────────────────────────
C vs B: p=0.032
B vs A: p=0.106 (not significant)
```

The contrastive mechanism significantly outperforms the specialized
prompt alone (p=0.032).

---

## Literature validation

Ground truth validation against published mechanisms — no evaluator
model, no metric. Binary: does the output contain confirmed key terms
from the published literature?

**Four conditions, 10 problems, N=3 each:**

```
Condition                       Score (key terms found)
────────────────────────────────────────────────────────
A: Generic prompt               0.60
B: Specialized prompt           0.80
C: Contrastive + specialized    0.93   ← BEST
D: Contrastive + generic        0.40   ← WORSE THAN BASELINE
────────────────────────────────────────────────────────
C vs A: p=0.030
C vs D: p=0.0012
```

Condition D confirms: contrastive with a generic prompt is significantly
worse than a generic prompt alone. The prompt provides direction;
the mechanism provides depth. Neither alone achieves what both produce.

---

## Install

```bash
pip install prism-synthesis
```

For token-level generation (required for the full mechanism):
```bash
pip install prism-synthesis[local]
# Requires: transformers, torch, GPU with 8GB+ VRAM
# Tested: Mistral-7B-Instruct, Qwen2.5-3B-Instruct
```

---

## Quick start

### Token-level (local GPU — primary mechanism)

```python
from prism import TokenLevelPRISM, PRISMConfig, get_mode

prism = TokenLevelPRISM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    config=PRISMConfig(alpha=1.5),
)

mode = get_mode("synthesis")
result = prism.generate(
    problem="Why does sleep deprivation paradoxically improve depression?",
    creative_system=mode.creative_system,
    conservative_system=mode.conservative_system,
)
print(result.text)
```

### Response-level (API — simpler, no GPU needed)

```python
from prism import PRISM

engine = PRISM.from_env()  # auto-detects ANTHROPIC_API_KEY or OPENAI_API_KEY
result = engine.synthesize(
    problem="Why does sleep deprivation paradoxically improve depression?",
    domain_a="neuroscience",
    domain_b="information theory",
)
print(result.synthesis)
print(f"PSS: {result.pss.pss:.4f}")
```

---

## Reproduce the results

```bash
# The evaluator asymmetry finding (p=0.0021)
python experiments/evaluator_asymmetry.py

# The cognitive state map (alpha × task table)
python experiments/cognitive_modes.py

# The alpha sweep on synthesis tasks
python experiments/alpha_sweep.py

# Literature validation against ground truth
python experiments/literature_validation.py
```

Pre-computed results for all experiments are in `results/`.

---

## Available cognitive modes

```python
from prism import list_modes
print(list_modes())
```

- `synthesis` — scientific cross-domain structural transfer (alpha=1.5)
- `forensic` — anomaly and pattern detection (alpha=0.5)
- `uncertainty` — calibrated uncertainty surfacing (alpha=0.5)
- `temporal` — long-term systemic reasoning (alpha=0.5)
- `contradiction` — logical tension detection (alpha=1.0)
- `adversarial` — structural weakness identification (alpha=1.0)
- `depth` — mechanistic depth over breadth (alpha=1.5)
- `analogy` — deep cross-domain analogical mapping (alpha=2.0)

---

## Project structure

```
prism/
├── prism/
│   ├── core/
│   │   ├── token_level.py    # Token-level contrastive engine (primary)
│   │   └── modes.py          # Cognitive mode definitions
│   ├── engine.py             # Response-level API engine
│   ├── pss.py                # PSS metric (AD × coherence)
│   ├── evaluator/            # Evaluation prompts and coherence
│   └── providers/            # Anthropic, OpenAI, Ollama
├── experiments/              # Reproducible experiment scripts
├── examples/                 # Quick start and usage examples
├── results/                  # Pre-computed experiment outputs
└── tests/
```

---

## Citation

```bibtex
@software{prism2026,
  title={PRISM: Precision Reduction for Isomorphic Synthesis and Mapping},
  author={Manh\~{a}es, Gabriel},
  year={2026},
  url={https://github.com/gabrielmanhaes/prism}
}
```

## License

MIT
