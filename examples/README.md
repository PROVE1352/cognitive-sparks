# Examples

## Quick Start

### 1. Analyze Claude Code architecture (included data)

```bash
# Uses the demo data shipped with this repo — no API key needed to read results
sparks run --goal "Find the core architecture principles" --data ./demo/claude_code_posts/ --depth quick
```

**Pre-computed result**: [claude_code_analysis.md](claude_code_analysis.md) — 3 principles found, 83% confidence, $0.52 cost.

### 2. Use with Ollama (free, local)

```bash
# Start Ollama with any model
ollama serve
ollama pull llama3.1

# Run Sparks against local model
SPARKS_BACKEND=openai-compat \
SPARKS_COMPAT_BASE_URL=http://localhost:11434/v1 \
  sparks run --goal "Find the key patterns" --data ./your-data/ --depth quick
```

### 3. Python SDK

```python
from sparks.api import Sparks

s = Sparks(goal="Find core principles", depth="standard")
result = s.run("./my-data/")

for p in result.principles:
    print(f"[{p.confidence}%] {p.statement}")
    print(f"  Evidence: {', '.join(p.supporting_patterns)}")

print(f"Cost: ${result.cost:.2f}")
```

### 4. Full loop (validate + evolve)

```python
from sparks.api import Sparks

s = Sparks(goal="Understand market dynamics", depth="deep")

# Phase A: Extract principles
result = s.run("./market-data/")

# Phase B-E: Validate and evolve across cycles
loop_result = s.loop(
    data_path="./new-data/",
    cycles=3,
    predict="Next quarter: rate cut + earnings season",
    outcomes="What actually happened: rates held, tech earnings beat",
)

# Knowledge accumulates
wiki = s.wiki()
wiki.ingest("./daily-reports/")
answer = wiki.query("What drives semiconductor cycles?")
```

### 5. Ablation experiments

```bash
# Compare: neural circuit vs sequential pipeline
sparks run --goal "..." --data ./data/ --depth standard          # autonomic (default)
sparks run --goal "..." --data ./data/ --depth standard --no-nervous  # sequential

# Disable specific neuromodulators
sparks run --goal "..." --data ./data/ --ablate dopamine
sparks run --goal "..." --data ./data/ --ablate stdp
```

## Pre-computed Results

| Example | Data | Principles | Confidence | Cost |
|---|---|---|---|---|
| [Claude Code analysis](claude_code_analysis.md) | 3 architecture posts | 3 | 83% | $0.52 |

These let you see what Sparks output looks like without running anything.
