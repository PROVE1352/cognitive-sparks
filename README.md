<p align="center">
  <h1 align="center">Sparks</h1>
  <p align="center"><strong>13 cognitive primitives that teach AI to think — not just compute.</strong></p>
  <p align="center">
    <a href="https://pypi.org/project/cognitive-sparks/"><img src="https://img.shields.io/pypi/v/cognitive-sparks?color=blue" alt="PyPI"></a>
    <a href="https://github.com/PROVE1352/cognitive-sparks/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PROVE1352/cognitive-sparks" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
    <a href="https://github.com/PROVE1352/cognitive-sparks/wiki"><img src="https://img.shields.io/badge/docs-wiki-green" alt="Docs"></a>
  </p>
</p>

> Give it a goal and data. It observes, abstracts, analogizes, plays, and synthesizes — like a Renaissance polymath. No hardcoded pipeline. Tool execution order **emerges** from neural dynamics.

**[한국어](README_KR.md)**

---

## Results First

We fed Sparks 15 months of real-world market data (640K chars) and asked it to find core principles — **without telling it what to look for.**

| | Vanilla LLM | Sparks (7 tools) | Sparks (13 tools) |
|---|---|---|---|
| Principles found | 3 | 7 | **12** |
| Avg confidence | — | 80% | **91%** |
| Human-expert coverage | — | 68% | **85%** |
| Cost | $0.03 | $6 | **$9** |

The 13-tool deep analysis independently rediscovered 3 manually-extracted laws **plus** 9 additional principles that human experts hadn't formalized. [Full benchmark details](benchmarks/)

---

## Install

```bash
pip install cognitive-sparks
```

Or from source:
```bash
git clone https://github.com/PROVE1352/cognitive-sparks.git
cd cognitive-sparks && pip install -e ".[all]"
```

## 30-Second Demo

```bash
# Analyze any data — circuit decides which tools to fire and in what order
sparks run --goal "Find the core principles" --data ./your-data/ --depth standard

# Validate + evolve principles against new data
sparks loop -p output/results.md -d ./new-data/ --cycles 3

# Disable neural circuit (sequential baseline, for comparison)
sparks run --goal "..." --data ./your-data/ --no-nervous
```

**Depth modes:**

| Mode | Tools fired | Budget | Use case |
|---|---|---|---|
| `quick` | 4 core | ~$0.15 | Fast exploration |
| `standard` | 7 | ~$2-6 | Daily analysis |
| `deep` | All 13 | ~$5-15 | Deep research |

**Works with any LLM:** Claude (default), GPT-4o, Gemini, Ollama, Groq. See [Backends](#backends).

---

## How It Works

```
Orchestra (LangGraph/CrewAI):     Conductor tells musicians what to play, in order
Nervous System (Sparks):          No conductor. No fixed order.
                                  Tool execution EMERGES from neural dynamics.
```

### The 3-Layer Architecture

```
                    Goal + Data
                        |
        +===============|================+
        |  NEURAL CIRCUIT (Layer 0)      |
        |  ~30 LIF neurons, ~80 STDP    |
        |  connections, neuromodulation   |
        |  Tool order emerges from       |
        |  activation dynamics           |
        +===============|================+
        |  13 THINKING TOOLS (Layer 1)   |
        |  Fire when activated:          |
        |  observe, imagine, abstract,   |
        |  patterns, analogize, model,   |
        |  body_think, empathize,        |
        |  shift_dimension, play,        |
        |  transform, synthesize         |
        +===============|================+
        |  AI AUGMENTATION (Layer 2)     |
        |  strategic forgetting,         |
        |  cross-session STDP learning   |
        +===============|================+
                        |
            Principles + Evidence
            + Confidence + Analogies
```

### Emergent Tool Ordering

The pipeline `observe -> patterns -> abstract -> synthesize` is **NOT coded**. It emerges:

| State | Winner | Why |
|---|---|---|
| Empty (no data) | **observe** | `obs_hunger=1.0` drives observe neuron |
| 20 observations | **recognize_patterns** | `obs_count` rises, patterns hungry |
| + 8 patterns | **abstract** | `pat_count` rises, principles hungry |
| + 4 principles | **analogize** | principles drive analogy neuron |
| + analogies | **synthesize** | all inputs ready, integrate mode |

This sequence is learned via STDP and evolves across sessions.

---

## The 13 Thinking Tools

Based on [Sparks of Genius](https://en.wikipedia.org/wiki/Sparks_of_Genius) (Root-Bernstein, 1999) — the thinking tools used by Einstein, Picasso, da Vinci, and Feynman.

| # | Tool | What it does |
|---|---|---|
| 1 | **observe** | Extract observations through domain-specific channels |
| 2 | **imagine** | "What if X at 10x? What if opposite?" |
| 3 | **abstract** | Picasso Bull: strip non-essential until essence remains |
| 4 | **recognize_patterns** | Find recurring, absent, interference patterns |
| 5 | **form_patterns** | Combine patterns into emergent structures |
| 6 | **analogize** | Deep structural correspondence across domains |
| 7 | **body_think** | Feel the data: weight, texture, rhythm, tension |
| 8 | **empathize** | Become the actors inside the data |
| 9 | **shift_dimension** | Same data, different axis: time, scale, frequency |
| 10 | **model** | Build rough model, see what breaks |
| 11 | **play** | Remove/invert/exaggerate — systematic rule-breaking |
| 12 | **transform** | Narrative to equation, static to dynamic |
| 13 | **synthesize** | All tools resonate at once |

---

## Full Pipeline

```
Phase A: sparks run     Data -> 13 Tools -> Principles
Phase B: Validate       Principles + New Data -> support/contradict scores
Phase C: Evolve         Weak principles refined/dropped, strong kept
Phase D: Predict        Validated principles -> testable predictions
Phase E: Feedback       Predictions vs outcomes -> strengthen/weaken
Phase F: Self-Optimize  Analyze output -> surgical fixes -> re-run
```

```bash
# Generate predictions from validated principles
sparks loop -p output/results.md --predict "Next week: Fed meeting + tariff deadline"

# Compare predictions to reality
sparks loop -p output/results.md --predict "..." --outcomes "What actually happened: ..."
```

---

## Backends

| Backend | Provider | Setup |
|---|---|---|
| `cli` (default) | Claude Code CLI | Free with subscription |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI / GPT-4o | `OPENAI_API_KEY` |
| `google` | Google Gemini | `GOOGLE_API_KEY` |
| `openai-compat` | Ollama, Groq, etc. | `SPARKS_COMPAT_BASE_URL` |

```bash
# Ollama (free, local)
SPARKS_BACKEND=openai-compat SPARKS_COMPAT_BASE_URL=http://localhost:11434/v1 \
  sparks run --goal "..." --data ./path/

# GPT-4o
SPARKS_BACKEND=openai OPENAI_API_KEY=sk-... sparks run --goal "..." --data ./path/
```

---

## Neural Circuit Details

Not if-else rules pretending to be neurons. Actual neural dynamics.

<details>
<summary><strong>Neuron Model & Learning</strong></summary>

**Rate-coded Leaky Integrate-and-Fire:**
```
tau * dr/dt = -(r - baseline) + I_total * gain + noise
if r > threshold: fire, enter refractory period
```

**Architecture:** 4 layers — Sensory (11) -> Signal (5) -> Tool (13) -> Mode (2)

| Mechanism | What it does |
|---|---|
| **STDP** | Co-firing tools strengthen connections; timing matters |
| **Reward modulation** | Tool success/failure modulates STDP via dopamine |
| **Homeostatic plasticity** | Over-active populations shrink incoming weights |
| **Neuromodulation** | DA (reward), NE (arousal), ACh (learning rate) |
| **Hunger signals** | Missing data types drive relevant tools |
| **Persistence** | Weights saved to JSON, evolve across sessions |

</details>

<details>
<summary><strong>Autonomic Cascade</strong></summary>

```
while tool_activation > threshold:
  1. Circuit runs 5 ticks (signals propagate)
  2. Highest activation tool = winner
  3. should_run() local ganglion check
  4. FIRE: tool executes (LLM call)
  5. State mutates -> sensory re-encoding
  6. Dopamine signal (success/failure)
  7. STDP learning (weights update)
  8. Checkpoint saved (crash recovery)
  -> Next tick

Stops when: all tools below threshold OR budget exhausted
```

After first cascade exhausts: **Consolidation** (prune, merge, reset) -> second cascade from fresh perspective -> **Convergence** (principles found independently in both rounds = real).

</details>

<details>
<summary><strong>Self-Optimization (Phase F)</strong></summary>

```
1. DIAGNOSE    Analyze output -> per-tool quality score
2. GENERATE    Surgical prompt fixes + weight tuning
3. PLAY-TEST   Stress-test each fix:
               INVERT: would opposite be better?
               REMOVE: would STDP self-correct?
               EXAGGERATE: push 10x — what breaks?
4. APPLY       Only safe fixes (with backup)
```

</details>

---

## Project Structure

```
sparks/
├── circuit.py        # Neural circuit (LIF + STDP + neuromodulation)
├── autonomic.py      # Pulse-driven cascade (no hardcoded order)
├── engine.py         # Sequential pipeline (--no-nervous fallback)
├── state.py          # CognitiveState + signal types
├── loop.py           # Full loop: validate -> evolve -> predict -> feedback
├── meta.py           # Self-analysis + code improvement
├── wiki.py           # Persistent knowledge accumulation
├── explain.py        # Explainability traces
├── llm.py            # Multi-provider LLM backend
├── tools/            # 13 thinking tools
└── cli.py            # CLI interface
```

---

## Status

- [x] 13/13 thinking tools implemented
- [x] Neural circuit (rate-coded LIF, STDP, neuromodulation)
- [x] Autonomic cascade (emergent tool ordering)
- [x] Full loop (validate -> evolve -> predict -> feedback)
- [x] Self-optimization (meta-loop)
- [x] Cross-session learning (persistent weights)
- [x] Multi-model (Claude, GPT-4o, Gemini, Ollama, Groq)
- [x] A-Test validation (12 principles, 91% confidence)
- [x] Wiki knowledge base (persistent, cross-referenced)
- [x] Explainability traces (per-firing attribution)
- [ ] Embedding-based convergence
- [ ] Large dataset benchmarks

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

---

*Built with [Claude Code](https://claude.ai/code). The framework that thinks about thinking.*
