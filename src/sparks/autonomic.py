"""Autonomic Engine — pulse-driven cascade execution.

Replaces the sequential for-loop with circuit-driven emergent execution.

Key insight: we can't run tools truly in parallel (LLM calls are expensive),
but we CAN let the circuit decide WHICH tool fires next and WHEN to stop.

Architecture:
                    ┌─────────────────────────────────┐
                    │         Neural Circuit           │
                    │  (continuous tick, ~30 populations)│
                    │                                   │
                    │  sensory ──→ signal ──→ tool      │
                    │     ↑          ↑         │        │
                    │     │     lateral inhibition       │
                    │     │          │         ↓        │
                    │     └──── state update ◄── fire   │
                    └─────────────────────────────────┘
                                    │
                              Which tool fired?
                                    │
                              ┌─────┴─────┐
                              │  Execute   │
                              │ (LLM call) │
                              └─────┬─────┘
                                    │
                              State mutated
                                    │
                              ┌─────┴─────┐
                              │  Re-encode │
                              │  sensory   │
                              └─────┬─────┘
                                    │
                              Feed back into circuit
                              (next tick starts)
                                    │
                              Continue until:
                              - No tool above threshold
                              - Budget exhausted
                              - sufficient_depth fires

This is analogous to how the real nervous system works:
- Sensory input arrives
- Neural populations integrate
- The one that crosses threshold first → fires
- Firing changes the state (muscle contraction / tool execution)
- Changed state = new sensory input
- Cycle repeats
- No central controller decides the order
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console

from sparks.circuit import NeuralCircuit
from sparks.cost import CostTracker, DEPTH_BUDGETS, get_active_tools
from sparks.data import DataStore
from sparks.events import EventBus
from sparks.lens import bootstrap_lens, quick_scan, sense_domain
from sparks.configurator import ToolConfigurator, apply_config
from sparks.nervous import (
    sense, reflex_check, filter_by_prediction, consolidate,
    update_synapses, check_feedback_needed,
)
from sparks.output import format_output
from sparks.persistence import SessionMemory
from sparks.state import CognitiveState, Phase, SynthesisOutput
from sparks.tools import TOOL_REGISTRY

console = Console()

# Maximum tool firings per cascade (safety limit)
MAX_FIRINGS = 30
# Circuit ticks between tool firings (let signals propagate)
TICKS_PER_STEP = 5
# Minimum activation for a tool to fire
FIRE_THRESHOLD = 0.30


def run_autonomic(
    goal: str,
    data_path: str,
    depth: str = "standard",
) -> SynthesisOutput:
    """Autonomic execution: circuit drives everything.

    No TOOL_ORDER. No Phase 1/2 distinction.
    The circuit senses state → accumulates → fires tools → repeats.
    """
    budget = DEPTH_BUDGETS[depth]
    tracker = CostTracker(budget)
    bus = EventBus()
    memory = SessionMemory()

    state = CognitiveState(goal=goal, depth=depth)
    active_tool_names = get_active_tools(depth)
    state.signals.active_tools = active_tool_names

    # Instantiate tools
    tools = {}
    for name in active_tool_names:
        if name in TOOL_REGISTRY:
            tools[name] = TOOL_REGISTRY[name](event_bus=bus, tracker=tracker)

    # ── Neural Circuit ──
    circuit = NeuralCircuit()
    if circuit.load():
        console.print(f"   [dim]🧠 Loaded persistent circuit (t={circuit.time_step})[/]")
    else:
        console.print(f"   [dim]🧠 New circuit initialized[/]")

    # Load persistent synapses
    persistent_synapses = memory.start_session()
    if persistent_synapses:
        state.signals.synapses = persistent_synapses

    # ── Load Data ──
    console.print(f"\n[bold]📊 Loading data from[/] {data_path}")
    data = DataStore(data_path)
    console.print(f"   {data.total_items} files, ~{data.estimated_tokens():,} tokens")

    # ── Bootstrap Lens ──
    console.print("\n[bold]🔍 Bootstrapping lens...[/]")
    state.lens = bootstrap_lens(data, goal, tracker=tracker)
    console.print(f"   Domain: [cyan]{state.lens.domain}[/]")
    console.print(f"   Focus: {state.lens.focus_questions[0] if state.lens.focus_questions else 'general'}")

    # ── Adaptive Configuration ──
    import os
    if os.environ.get("SPARKS_ALL_OPUS") != "1":
        profile = quick_scan(data, tracker=tracker)
        domain = sense_domain(profile)
        configurator = ToolConfigurator()
        adaptive = configurator.configure(profile, domain, goal, depth, tracker=tracker)
        from sparks.cost import MODEL_ROUTING
        updated_routing = apply_config(adaptive, MODEL_ROUTING)
        for tool_name, model in updated_routing.items():
            tracker._routing_overrides = getattr(tracker, '_routing_overrides', {})
            tracker._routing_overrides[tool_name] = model

    # ── Reflex Check (fast path, no LLM) ──
    reflexes = reflex_check(state)
    for r in reflexes:
        console.print(f"   [red]⚡ Reflex: {r}[/]")
        if r.startswith("STOP:"):
            return SynthesisOutput()

    # ════════════════════════════════════════════
    #  AUTONOMIC CASCADE — the core loop
    # ════════════════════════════════════════════

    console.print(f"\n[bold]⚡ Autonomic cascade starting[/]")
    output = SynthesisOutput()
    firings = 0
    firing_log: list[str] = []
    consolidation_done = False

    while firings < MAX_FIRINGS:
        # 1. Encode current state as sensory input
        sensory = NeuralCircuit.encode_state(state)

        # 2. Run circuit for several ticks (let signals propagate)
        for _ in range(TICKS_PER_STEP):
            circuit.update(sensory, dt=0.3)

        # 3. Check termination conditions
        if circuit.get_fired("sufficient"):
            console.print(f"   [green]✅ sufficient_depth fired → stopping[/]")
            break

        if not tracker.can_afford(tracker.select_model("synthesize")):
            console.print(f"   [yellow]⚠️ Budget exhausted[/]")
            break

        # 4. Find the tool with highest activation above threshold
        activations = circuit.get_tool_activations()
        # Filter to tools we actually have
        candidates = {
            name: rate for name, rate in activations.items()
            if name in tools and rate >= FIRE_THRESHOLD
        }

        if not candidates:
            # No tool above threshold — cascade exhausted naturally
            if not consolidation_done and state.principles:
                # Try consolidation (sleep) and re-ignite
                console.print(f"   [dim]😴 No tool above threshold → consolidation[/]")
                _do_consolidation(state, circuit)
                consolidation_done = True
                continue
            else:
                console.print(f"   [dim]⚡ Cascade exhausted ({firings} firings)[/]")
                break

        # 5. Winner-take-all: highest activation fires
        winner = max(candidates, key=candidates.get)
        activation = candidates[winner]

        # 6. Check should_run (local ganglion decision)
        tool = tools[winner]
        if hasattr(tool, 'should_run') and not tool.should_run(state):
            # Tool's local rule says no — suppress it temporarily
            circuit.populations[winner].refractory = 2.0  # Cool it down
            continue

        # 7. FIRE! Execute the tool
        firings += 1
        mode = circuit.get_mode()
        mode_icon = {"sympathetic": "🔴", "parasympathetic": "🔵", "balanced": "⚪"}.get(mode, "⚪")
        console.print(
            f"   {mode_icon} [{firings:2d}] {winner} "
            f"(activation={activation:.2f}, mode={mode})"
        )

        # Track state before execution (for Hebbian learning)
        pre_counts = {
            "observations": len(state.observations),
            "patterns": len(state.patterns),
            "principles": len(state.principles),
            "analogies": len(state.analogies),
        }

        try:
            kwargs = {}
            if winner in ("observe", "model", "body_think"):
                kwargs["data"] = data

            result = tool.run(state, **kwargs)

            if winner == "synthesize" and result is not None:
                output = result

            # Hebbian: did the tool produce new output?
            had_new = (
                (winner == "observe" and len(state.observations) > pre_counts["observations"]) or
                (winner == "recognize_patterns" and len(state.patterns) > pre_counts["patterns"]) or
                (winner == "abstract" and len(state.principles) > pre_counts["principles"]) or
                (winner == "analogize" and len(state.analogies) > pre_counts["analogies"]) or
                (winner == "synthesize" and result is not None) or
                (winner in ("body_think", "shift_dimension", "transform") and
                 len(state.observations) > pre_counts["observations"]) or
                (winner in ("imagine", "play", "empathize", "form_patterns"))  # Always "new"
            )

            # 8. Feed outcome back into circuit (dopamine signal)
            circuit.record_tool_outcome(winner, success=had_new)
            update_synapses(state, winner, success=had_new)

            cost = tracker.breakdown.get(winner, 0)
            console.print(f"         → {'✓ new output' if had_new else '· no change'} (${cost:.3f})")

        except Exception as e:
            console.print(f"         → [red]✗ {e}[/]")
            circuit.record_tool_outcome(winner, success=False)
            update_synapses(state, winner, success=False)

        # 9. Post-fire: tool enters refractory (won't fire again immediately)
        circuit.populations[winner].refractory = circuit.populations[winner].refractory_period

        firing_log.append(winner)

        # 10. Update nervous system signals (for convergence detection etc.)
        state.signals = sense(state)

    # ════════════════════════════════════════════
    #  POST-CASCADE
    # ════════════════════════════════════════════

    # Ensure output has principles
    if not output.principles and state.principles:
        output.principles = state.principles
    output.total_cost = tracker.total_cost
    output.rounds_completed = firings
    output.tools_used = list(set(firing_log))
    output.contradictions = state.contradictions
    output.analogies = output.analogies or state.analogies

    # Save everything
    memory.end_session(state, output)
    circuit.save()

    # ── Report ──
    console.print(f"\n[bold]═══ Cascade Complete ═══[/]")
    console.print(f"Firings: {firings} | Tools used: {len(set(firing_log))}")
    console.print(f"Firing sequence: {' → '.join(firing_log)}")
    console.print(f"Mode: {circuit.get_mode()}")
    console.print(f"\n{circuit.status()}")
    console.print(f"\n[bold]💰 Cost:[/] ${tracker.total_cost:.2f} / ${budget.max_cost:.2f}")
    console.print(f"[bold]📋 {len(state.principles)} principles[/] (avg confidence: {_avg_conf(state):.0%})")

    return output


def _do_consolidation(state: CognitiveState, circuit: NeuralCircuit):
    """Sleep-like consolidation: prune, merge, then boost circuit for fresh pass."""
    result = consolidate(state)
    console.print(f"         Pruned {result['pruned_observations']} obs, merged {result['merged_patterns']} patterns")

    # Reset circuit tool populations (fresh start after sleep)
    from sparks.circuit import TOOLS
    for name in TOOLS:
        if name in circuit.populations:
            pop = circuit.populations[name]
            pop.rate = pop.baseline  # Reset to baseline
            pop.refractory = 0.0     # Clear refractory

    # Boost NE (wake up, explore)
    circuit.norepinephrine = min(1.0, circuit.norepinephrine + 0.3)

    state.clean_slate()
    state.phase = Phase.ITERATIVE


def _avg_conf(state: CognitiveState) -> float:
    if not state.principles:
        return 0.0
    return sum(p.confidence for p in state.principles) / len(state.principles)
