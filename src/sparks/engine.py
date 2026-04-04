"""Cognitive Engine — orchestrates the full pipeline."""

from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sparks.configurator import ToolConfigurator, apply_config, AdaptiveConfig
from sparks.cost import CostTracker, DEPTH_BUDGETS, MODEL_ROUTING, get_active_tools
from sparks.data import DataStore
from sparks.events import EventBus
from sparks.lens import bootstrap_lens, quick_scan, sense_domain
from sparks.nervous import (
    sense, should_continue, reflex_check, filter_by_prediction,
    competitive_tool_selection, update_synapses, check_feedback_needed,
    consolidate, boost_rhythm_group, record_signal_contributor,
)
from sparks.circuit import NeuralCircuit
from sparks.output import format_output
from sparks.persistence import SessionMemory
from sparks.state import CognitiveState, Phase, SynthesisOutput
from sparks.tools import TOOL_REGISTRY

console = Console()

# Execution order for Phase 1 (sequential)
TOOL_ORDER = [
    "observe",
    "body_think",
    "shift_dimension",
    "recognize_patterns",
    "form_patterns",
    "empathize",
    "abstract",
    "analogize",
    "imagine",
    "model",
    "play",
    "transform",
    "synthesize",
]


def run(goal: str, data_path: str, depth: str = "standard", nervous_system: bool = True) -> SynthesisOutput:
    """Main entry point. Goal + Data → Core Principles.

    nervous_system=False: ablation mode — simple pipeline, no biological signals.
    """

    budget = DEPTH_BUDGETS[depth]
    tracker = CostTracker(budget)
    bus = EventBus()
    memory = SessionMemory()

    state = CognitiveState(goal=goal, depth=depth)

    # ── Neural Circuit (real nervous system) ──
    circuit = None
    if nervous_system:
        circuit = NeuralCircuit()
        if circuit.load():
            console.print(f"   [dim]🧠 Loaded persistent circuit (t={circuit.time_step})[/]")
        else:
            console.print(f"   [dim]🧠 Initialized new neural circuit[/]")

    # Load persistent synapses from past sessions
    persistent_synapses = memory.start_session()
    if persistent_synapses:
        state.signals.synapses = persistent_synapses
        console.print(f"   [dim]🧠 Loaded {len(persistent_synapses)} persistent synapses[/]")
    active = get_active_tools(depth)
    state.signals.active_tools = active

    # Instantiate tools
    tools = {}
    for name in TOOL_ORDER:
        if name in active and name in TOOL_REGISTRY:
            tools[name] = TOOL_REGISTRY[name](event_bus=bus, tracker=tracker)

    # ── Load Data ──
    console.print(f"\n[bold]📊 Loading data from[/] {data_path}")
    data = DataStore(data_path)
    console.print(f"   {data.total_items} files, ~{data.estimated_tokens():,} tokens")

    # ── Bootstrap Lens ──
    console.print("\n[bold]🔍 Bootstrapping lens...[/]")
    state.lens = bootstrap_lens(data, goal, tracker=tracker)
    console.print(f"   Domain: [cyan]{state.lens.domain}[/]")
    console.print(f"   Channels: {', '.join(ch.name for ch in state.lens.channels)}")
    console.print(f"   Focus: {state.lens.focus_questions[0] if state.lens.focus_questions else 'general'}")

    # ── Adaptive Configuration ──
    if nervous_system:
        profile = quick_scan(data, tracker=tracker)
        domain = sense_domain(profile)
        configurator = ToolConfigurator()
        adaptive = configurator.configure(profile, domain, goal, depth, tracker=tracker)

        import os
        if os.environ.get("SPARKS_ALL_OPUS") != "1":
            updated_routing = apply_config(adaptive, MODEL_ROUTING)
            for tool_name, model in updated_routing.items():
                tracker._routing_overrides = getattr(tracker, '_routing_overrides', {})
                tracker._routing_overrides[tool_name] = model
        else:
            console.print("   [dim]SPARKS_ALL_OPUS=1: skipping adaptive model overrides[/]")

        if adaptive.model_overrides:
            console.print(f"   [dim]Model overrides: {adaptive.model_overrides}[/]")
        if adaptive.external_suggestions:
            console.print(f"   [dim]💡 Suggestions: {adaptive.external_suggestions[0]}[/]")
        if adaptive.nervous_hints:
            for hint in adaptive.nervous_hints:
                console.print(f"   [yellow]⚡ {hint}[/]")

        state._tool_hints = {k: v.model_dump() for k, v in adaptive.tool_configs.items()}

        reflexes = reflex_check(state)
        for r in reflexes:
            console.print(f"   [red]⚡ Reflex: {r}[/]")
            if r.startswith("STOP:"):
                return SynthesisOutput()
    else:
        console.print("   [yellow]⚠️ Nervous system DISABLED (ablation mode)[/]")

    # ── Phase 1: Sequential ──
    output = _run_round(state, tools, data, tracker, "Phase 1", circuit=circuit)

    # ── Phase 2: Iterate (standard/deep only) ──
    if depth != "quick" and should_continue(state, budget.max_rounds):
        # Predictive coding: Round 1 principles become predictions for Round 2
        round1_predictions = [p.statement for p in state.principles]

        if nervous_system:
            console.print("\n[bold]😴 Consolidation (sleep-like cleanup)...[/]")
            result_info = consolidate(state)
            console.print(f"   Pruned {result_info['pruned_observations']} obs, merged {result_info['merged_patterns']} patterns")
            synapse_preview = dict(list(state.signals.synapses.items())[:3])
            if synapse_preview:
                console.print(f"   Synapses: {synapse_preview}...")
        else:
            console.print("\n[bold]🧹 Clean slate (simple reset)...[/]")

        state.clean_slate()
        state.phase = Phase.ITERATIVE

        # Store predictions for use in observe tool
        state._predictions = round1_predictions

        phase2_output = _run_round(state, tools, data, tracker, "Phase 2", circuit=circuit)
        # Keep Phase 2 output only if synthesize ran; otherwise preserve Phase 1
        if phase2_output.principles:
            output = phase2_output

        # Prediction error reporting
        surprising, predicted = filter_by_prediction(
            state.observations, round1_predictions
        )
        state.signals.prediction_errors = [
            o.content[:100] for o in surprising[:5]
        ]
        if surprising:
            console.print(f"   [cyan]🧠 Prediction errors: {len(surprising)} surprising observations[/]")

        # Convergence check
        state.signals = sense(state)
        if circuit:
            sensory = NeuralCircuit.encode_state(state)
            circuit.update(sensory)
            console.print(f"   [dim]{circuit.status().split(chr(10))[0]}[/]")
        if state.signals.convergence:
            console.print("[green]✅ Convergence detected[/]")
        else:
            console.print("[yellow]⚠️ Not fully converged[/]")

    # ── Final stats ──
    state.signals.total_cost = tracker.total_cost

    # Ensure output has principles from state (in case synthesize was skipped)
    if not output.principles and state.principles:
        output.principles = state.principles
    output.total_cost = tracker.total_cost
    output.rounds_completed = state.round + 1
    output.tools_used = state.signals.active_tools
    output.contradictions = state.contradictions
    output.analogies = output.analogies or state.analogies

    # Save session (synapses + knowledge base + circuit)
    memory.end_session(state, output)
    if circuit:
        circuit.save()
        console.print(f"   [dim]🧠 Circuit saved (t={circuit.time_step})[/]")

    console.print(f"\n[bold]💰 Total cost:[/] ${tracker.total_cost:.2f} / ${budget.max_cost:.2f}")
    console.print(f"[bold]📋 {len(state.principles)} principles[/] (avg confidence: {_avg_conf(state):.0%})")
    if memory.kb.total_entries > 0:
        console.print(f"[dim]🧠 Knowledge base: {memory.kb.total_entries} past analyses, {len(memory.kb.domains_seen)} domains[/]")

    return output


def _run_round(
    state: CognitiveState,
    tools: dict,
    data: DataStore,
    tracker: CostTracker,
    phase_label: str,
    circuit: NeuralCircuit = None,
) -> SynthesisOutput:
    """Execute one round of all active tools."""
    output = SynthesisOutput()

    # ── Circuit-based tool selection (if circuit available) ──
    if circuit and state.phase != Phase.SEQUENTIAL:
        # Feed current state to circuit
        sensory = NeuralCircuit.encode_state(state)
        for _ in range(3):  # Run 3 update steps for signal propagation
            circuit.update(sensory, dt=0.5)

        active_names = circuit.get_active_tools(threshold=0.25)
        # Always include synthesize
        if "synthesize" not in active_names and "synthesize" in tools:
            active_names.append("synthesize")
        # Only keep tools we actually have
        active_names = [n for n in active_names if n in tools]
        console.print(f"   [dim]🧠 Circuit selection ({circuit.get_mode()}): {active_names}[/]")
    elif state.phase != Phase.SEQUENTIAL and len(tools) > 4:
        # Fallback: old lateral inhibition
        rhythm_boosts = boost_rhythm_group(state, tools)
        for name, boost in rhythm_boosts.items():
            state.signals.tool_rewards[name] = state.signals.tool_rewards.get(name, 0) + boost

        active_names = competitive_tool_selection(tools, state, max_active=min(len(tools), 6))
        console.print(f"   [dim]🧠 Lateral inhibition + rhythm: {active_names}[/]")
    else:
        active_names = list(tools.keys())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[bold]{phase_label}[/] Round {state.round + 1}", total=len(active_names))

        for name in active_names:
            tool = tools.get(name)
            if not tool:
                progress.advance(task)
                continue

            progress.update(task, description=f"[bold]{phase_label}[/] {name}")

            if not tracker.can_afford(tracker.select_model(name)):
                console.print(f"   [yellow]⚠️ Budget low, skipping {name}[/]")
                progress.advance(task)
                continue

            # Distributed control: tool decides locally if it should run
            if hasattr(tool, 'should_run') and not tool.should_run(state):
                # In Phase 1, override: run anyway (training wheels)
                if state.phase != Phase.SEQUENTIAL:
                    progress.console.print(f"   [dim]{name}: skipped (should_run=False)[/]")
                    progress.advance(task)
                    continue

            # Track tool activity (homeostasis) + firing order (STDP)
            state.signals.tool_activity[name] = state.signals.tool_activity.get(name, 0) + 1

            try:
                kwargs = {}
                if name in ("observe", "model", "body_think"):
                    kwargs["data"] = data

                # Track counts before run for delta-based Hebbian learning
                pre_counts = {
                    "observations": len(state.observations),
                    "patterns": len(state.patterns),
                    "principles": len(state.principles),
                    "analogies": len(state.analogies),
                }

                result = tool.run(state, **kwargs)

                if name == "synthesize" and result is not None:
                    output = result

                cost_so_far = tracker.breakdown.get(name, 0)
                progress.console.print(f"   {name}: ${cost_so_far:.3f}")

                # Hebbian plasticity: delta-based success detection
                had_new_output = (
                    (name == "observe" and len(state.observations) > pre_counts["observations"]) or
                    (name == "recognize_patterns" and len(state.patterns) > pre_counts["patterns"]) or
                    (name == "abstract" and len(state.principles) > pre_counts["principles"]) or
                    (name == "analogize" and len(state.analogies) > pre_counts["analogies"]) or
                    (name == "synthesize" and result is not None)
                )
                update_synapses(state, name, success=had_new_output)
                if circuit:
                    circuit.record_tool_outcome(name, success=had_new_output)

            except Exception as e:
                console.print(f"   [red]❌ {name} failed: {e}[/]")
                update_synapses(state, name, success=False)
                if circuit:
                    circuit.record_tool_outcome(name, success=False)

            progress.advance(task)

    # ── Recurrent feedback: re-activate tools if needed ──
    feedback = check_feedback_needed(state)
    if feedback:
        for fb_tool_name in feedback[:2]:  # max 2 re-activations
            if fb_tool_name in tools and tracker.can_afford(tracker.select_model(fb_tool_name)):
                console.print(f"   [cyan]🔄 Feedback: re-activating {fb_tool_name}[/]")
                try:
                    kwargs = {}
                    if fb_tool_name in ("observe", "model", "body_think"):
                        kwargs["data"] = data
                    tools[fb_tool_name].run(state, **kwargs)
                    update_synapses(state, fb_tool_name, success=True)
                except Exception:
                    pass

    # Update signals after round
    state.signals = sense(state)

    return output


def _avg_conf(state: CognitiveState) -> float:
    if not state.principles:
        return 0.0
    return sum(p.confidence for p in state.principles) / len(state.principles)
