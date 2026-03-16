# PRISM

**Precision Reduction for Isomorphic Synthesis and Mapping**

Find structural equivalences between distant scientific domains. Given a problem, PRISM searches for formal frameworks in other fields that transfer entire problem-solving machineries — theorems, measurements, interventions — into your domain.

## How it works

PRISM uses a **contrastive plasticity mechanism**:
1. **Creative pass** — generate from a low-precision approximation (high temperature + creative prompting)
2. **Conservative pass** — generate a conventional response
3. **Delta extraction** — isolate what the creative pass adds beyond convention
4. **PSS evaluation** — score the result on attractor distance x structural coherence

The Precision-Survival Score (PSS) measures whether the output is both genuinely novel (high attractor distance) and structurally valid (high coherence evaluated by an independent model).

## Installation

```bash
# With Anthropic (Claude)
pip install prism-synthesis[anthropic]

# With OpenAI
pip install prism-synthesis[openai]

# With local models (Ollama)
pip install prism-synthesis[ollama]

# All providers
pip install prism-synthesis[all]
```

## Quick Start

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

from prism import PRISM

engine = PRISM.from_env()

result = engine.synthesize(
    problem="Why does sleep deprivation paradoxically improve depression?",
    domain_a="neuroscience",
    domain_b="information theory",
)

print(f"PSS: {result.pss.pss:.4f}")
print(result.synthesis)
```

## Provider Configuration

```python
from prism import PRISM, PRISMConfig
from prism.providers import AnthropicProvider, OpenAIProvider, OllamaProvider

# Anthropic Claude (generator) + local Ollama (evaluator — keeps costs low)
engine = PRISM(
    generator=AnthropicProvider(model="claude-sonnet-4-20250514"),
    evaluator=OllamaProvider(model="gemma3:4b"),
    config=PRISMConfig(min_pss=0.04, max_attempts=6, verbose=True)
)

# OpenAI only
engine = PRISM(
    generator=OpenAIProvider(model="gpt-4o"),
    config=PRISMConfig(min_pss=0.04)
)

# Fully local (free, slower)
engine = PRISM(
    generator=OllamaProvider(model="mistral"),
    evaluator=OllamaProvider(model="gemma3:4b"),
)
```

## Multi-Domain Search

```python
results = engine.search(
    problem="How do antibiotic-resistant bacteria always defeat targeted treatments?",
    domain_a="microbiology",
    target_domains=["cybernetics", "evolutionary biology", "game theory"],
    top_k=3,
)

for r in results:
    print(f"{r.domain_a} x {r.domain_b}: PSS={r.pss.pss:.4f}")
    print(r.synthesis[:200])
```

## Theoretical Background

PRISM implements the contrastive plasticity mechanism described in:

> Manhães, G. (2026). Precision-Survival Score: A Cross-Domain Structural Coherence Metric for Creative Synthesis. *bioRxiv* [link]

The mechanism is grounded in the precision dynamics framework for computational psychiatry:

> Manhães, G. (2026). A Unified Precision Dynamics Framework for Computational Psychiatry. *bioRxiv* [link]

The core theoretical claim: the contrastive mechanism approximates the same precision reduction that produces therapeutic effects in psychedelic-assisted therapy and generative insight in hypnagogic states. Creativity and psychiatric treatment are expressions of the same underlying computational operation.

## Citation

```bibtex
@software{prism2026,
  title={PRISM: Precision Reduction for Isomorphic Synthesis and Mapping},
  author={Manhães, Gabriel},
  year={2026},
  url={https://github.com/gabrielmanhaes/prism}
}
```

## License

MIT
