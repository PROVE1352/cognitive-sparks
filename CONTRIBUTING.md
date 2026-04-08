# Contributing to Sparks

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/PROVE1352/cognitive-sparks.git
cd cognitive-sparks
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Running

```bash
# Quick test with included demo data (Claude Code CLI, no API key needed)
sparks run --goal "Find core principles" --data ./demo/claude_code_posts/ --depth quick

# Ollama (free, local)
SPARKS_BACKEND=openai-compat SPARKS_COMPAT_BASE_URL=http://localhost:11434/v1 \
  sparks run --goal "Find core principles" --data ./demo/claude_code_posts/ --depth quick

# Anthropic API
ANTHROPIC_API_KEY=sk-ant-... SPARKS_BACKEND=anthropic \
  sparks run --goal "Find core principles" --data ./demo/claude_code_posts/ --depth standard
```

## Project Structure

```
src/sparks/
├── api.py             # Public API (Sparks class)
├── cli.py             # CLI entry point
├── state.py           # Core data models (start here)
├── circuit.py         # Neural circuit (LIF neurons, STDP, neuromodulation)
├── autonomic.py       # Autonomic cascade (emergent tool ordering)
├── engine.py          # Legacy sequential pipeline (--no-nervous fallback)
├── nervous.py         # Signal computation
├── loop.py            # Full loop: validate -> evolve -> predict -> feedback
├── meta.py            # Self-analysis + code improvement
├── wiki.py            # Persistent knowledge base
├── explain.py         # Explainability traces
├── lens.py            # Observation lens bootstrapping
├── configurator.py    # Adaptive domain routing
├── persistence.py     # Cross-session learning
├── evolution.py       # AutoAgent-style evolution
├── llm.py             # Multi-provider LLM backend
├── cost.py            # Model routing + budget tracking
├── context.py         # Per-tool context assembly
├── data.py            # Data loading + chunking
├── checkpoint.py      # Crash recovery
├── events.py          # Event bus
├── output.py          # Markdown output formatter
├── research.py        # Benchmarking utilities
└── tools/             # 13 thinking tools (all implemented)
    ├── base.py            # BaseTool interface
    ├── observe.py         # #1: Observation with lens
    ├── imagine.py         # #2: Mental simulation
    ├── abstract.py        # #3: Picasso Bull abstraction
    ├── patterns.py        # #4-5: Recognize + form patterns
    ├── analogize.py       # #6: Structural analogy
    ├── body_think.py      # #7: Feel the data physically
    ├── empathize.py       # #8: Become the actors inside
    ├── shift_dimension.py # #9: Change the axis
    ├── model_tool.py      # #10: Cardboard model testing
    ├── play.py            # #11: Systematic rule-breaking
    ├── transform.py       # #12: Representation change
    └── synthesize.py      # #13: Final integration
```

## Contribution Areas

### Easiest: Add a Domain Template

Edit `src/sparks/configurator.py` → `DOMAIN_CONFIGS`:

```python
"your_domain": {
    "tool_hints": {
        "observe": {"hint": "What to focus on in this domain"},
        "abstract": {"hint": "What kind of principles to extract"},
    },
    "model_overrides": {"abstract": "opus"},
    "tool_boost": ["empathize", "play"],
    "external_suggestions": ["Useful API or data source"],
}
```

### Medium: Add Benchmarks

We need more cross-domain benchmarks. See `benchmarks/` for the pattern:

1. Create `benchmarks/your_domain/data/` with source text files
2. Define expected themes in the benchmark runner
3. Run `python benchmarks/run_all.py --domains your_domain`
4. Compare Sparks vs CoT baseline

### Advanced: Improve the Neural Circuit

The circuit lives in `circuit.py`. Key areas:

- **Embedding-based convergence** — replace keyword matching in convergence detection
- **New neuromodulators** — serotonin (patience), oxytocin (trust in patterns)
- **Visualization** — real-time dashboard showing neuron activations during cascade

### How to Add a New Thinking Tool

All 13 core tools are implemented. But if you want to add a 14th:

```python
# 1. Create src/sparks/tools/your_tool.py
from sparks.tools.base import BaseTool
from sparks.state import CognitiveState

class YourTool(BaseTool):
    name = "your_tool"

    def should_run(self, state: CognitiveState) -> bool:
        """Local ganglion: when should this tool fire?"""
        return True  # your condition

    def run(self, state: CognitiveState, **kwargs):
        # 1. Read what you need from state
        # 2. Call LLM with structured output
        # 3. Write results to state
        # 4. Emit events
        pass

# 2. Register in src/sparks/tools/__init__.py
# 3. Add to TOOL_PRIORITY in src/sparks/cost.py
# 4. Add to TOOL_DATA_NEEDS in src/sparks/context.py
# 5. Add neural population in circuit.py
```

## What We Need Help With

- [ ] **More domain templates** — Legal, medical, scientific, educational...
- [ ] **Cross-domain benchmarks** — Sparks vs GPT-Researcher vs STORM on public datasets
- [ ] **Embedding-based convergence** — Better than keyword matching
- [ ] **Visualization dashboard** — Real-time nervous system state
- [ ] **Test suite** — pytest for circuit, tools, loop
- [ ] **Jupyter notebooks** — Interactive walkthroughs

## Guidelines

- **Keep tools autonomous.** Each tool decides when to fire via `should_run()`. No central ordering.
- **Structured output.** All LLM calls use `llm_structured()` with Pydantic schemas.
- **Cost awareness.** Track every LLM call through `CostTracker`.
- **Neural dynamics matter.** The circuit isn't decoration — it's architecture. New features should respect signal accumulation, threshold firing, and distributed control.
- **Absence is signal.** What's missing is often what matters most.

## Philosophy

The 13 thinking tools come from [Sparks of Genius](https://en.wikipedia.org/wiki/Sparks_of_Genius) (Root-Bernstein, 1999) — derived from studying how Einstein, Picasso, da Vinci, and Feynman actually thought.

The neural circuit is inspired by real neuroscience (LIF neurons, STDP plasticity, neuromodulation), not metaphor. When adding a feature, ask: "How does the real nervous system handle this?"

---

*Built with [Claude Code](https://claude.ai/code).*
