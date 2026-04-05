"""Neural Circuit — biologically-grounded nervous system.

Replaces if-else rules with actual neural dynamics.
Rate-coded Leaky Integrate-and-Fire populations with:
- Weighted connections (excitatory/inhibitory)
- STDP learning (spike-timing dependent plasticity)
- Neuromodulatory gain control
- Homeostatic plasticity
- Lateral inhibition via inhibitory interneurons

Behavior EMERGES from connection weights, not from if-else rules.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ─── Neural Population ───


class NeuralPopulation(BaseModel):
    """A population of neurons with shared firing rate (rate coding).

    drate/dt = (-(rate - baseline) + I_total) / tau
    fires when rate > threshold, then enters refractory period.
    """
    name: str
    rate: float = 0.0               # Firing rate (0.0 - 1.0)
    tau: float = 1.0                # Time constant (higher = more inertia)
    threshold: float = 0.5          # Firing threshold
    baseline: float = 0.0           # Resting rate
    refractory: float = 0.0         # Refractory timer (counts down)
    refractory_period: float = 0.3  # Duration of refractory after firing
    fired: bool = False             # Did this population fire this step?
    last_fire_time: int = -100      # When it last fired (for STDP)
    gain: float = 1.0               # Neuromodulatory gain multiplier

    def step(self, input_current: float, dt: float = 1.0, noise: float = 0.02):
        """One time step of leaky integration."""
        # Refractory: rate decays, no input accepted
        if self.refractory > 0:
            self.refractory -= dt
            self.rate *= (1.0 - 0.3 * dt)  # Fast decay during refractory
            self.fired = False
            return

        # Stochastic resonance: small noise helps weak signals cross threshold
        noise_val = random.gauss(0, noise) if noise > 0 else 0.0

        # Leaky integration: tau * dr/dt = -(r - baseline) + I * gain
        total_input = input_current * self.gain + noise_val
        dr = (-(self.rate - self.baseline) + total_input) / max(self.tau, 0.1)
        self.rate = max(0.0, min(1.0, self.rate + dr * dt))

        # Threshold check
        if self.rate >= self.threshold:
            self.fired = True
            self.refractory = self.refractory_period
        else:
            self.fired = False


# ─── Connection (Synapse) ───


class Connection(BaseModel):
    """Weighted connection between two neural populations."""
    source: str
    target: str
    weight: float = 0.5        # -1.0 (strong inhibition) to 1.0 (strong excitation)
    plasticity: bool = True    # Can this connection learn?
    sign: int = 1              # 1 = excitatory, -1 = inhibitory (Dale's law)

    @property
    def effective_weight(self) -> float:
        return self.weight * self.sign


# ─── Neural Circuit ───


SENSORY = [
    "obs_count", "pat_count", "prin_count", "contra_count",
    "failure_count", "round_num", "confidence_avg", "cost_ratio",
    "obs_hunger", "pat_hunger", "prin_hunger",  # Absence = signal (like hunger)
]

SIGNALS = [
    "convergence", "contradiction", "diminishing", "anomaly", "sufficient",
]

TOOLS = [
    "observe", "imagine", "recognize_patterns", "form_patterns",
    "abstract", "analogize", "body_think", "empathize",
    "shift_dimension", "model", "play", "transform", "synthesize",
]

MODES = ["explore", "integrate"]

ALL_POPULATIONS = SENSORY + SIGNALS + TOOLS + MODES


class NeuralCircuit(BaseModel):
    """The full nervous system as a neural circuit.

    ~30 neural populations connected by weighted synapses.
    No if-else rules. Behavior emerges from dynamics.
    """
    populations: dict[str, NeuralPopulation] = {}
    connections: list[Connection] = []
    time_step: int = 0

    # Neuromodulators (global gain control)
    dopamine: float = 0.0       # Reward signal: modulates plasticity magnitude
    norepinephrine: float = 0.5 # Arousal: modulates all gains
    acetylcholine: float = 0.5  # Learning rate: modulates STDP window

    # STDP parameters
    stdp_lr: float = 0.01       # Base learning rate
    stdp_window: int = 3        # Time steps for STDP correlation

    # Homeostatic Learning Rate (Kingson Man et al., "Need is All You Need")
    # Each tool's lr self-adjusts based on its own performance.
    # Good performance → lr decreases (exploit: stable weights)
    # Bad performance → lr increases (explore: rapid relearning)
    # Domain shift → all lr spike (fast unlearning + relearning)
    tool_lr: dict[str, float] = {}       # Per-tool adaptive learning rate
    tool_performance: dict[str, float] = {}  # Exponential moving avg of success
    lr_min: float = 0.003
    lr_max: float = 0.05
    domain_shift_detected: bool = False

    # Homeostatic target
    target_rate: float = 0.3    # Desired average firing rate

    # Ablation flags (disable individual components for experiments)
    ablate_dopamine: bool = False
    ablate_norepinephrine: bool = False
    ablate_acetylcholine: bool = False
    ablate_stdp: bool = False
    ablate_homeostatic: bool = False

    def model_post_init(self, __context):
        if not self.populations:
            self._init_populations()
        if not self.connections:
            self._init_connections()

    def _init_populations(self):
        """Create all neural populations with biologically-inspired parameters."""
        # Sensory: fast, low threshold (responsive to input)
        for name in SENSORY:
            self.populations[name] = NeuralPopulation(
                name=name, tau=0.3, threshold=0.1, refractory_period=0.0,
            )

        # Signal: medium speed, medium threshold (integrators)
        thresholds = {
            "convergence": 0.55, "contradiction": 0.35, "diminishing": 0.45,
            "anomaly": 0.4, "sufficient": 0.6,
        }
        for name in SIGNALS:
            self.populations[name] = NeuralPopulation(
                name=name, tau=1.2, threshold=thresholds.get(name, 0.5),
                refractory_period=0.5,
            )

        # Tools: slower, variable thresholds (each tool "decides" locally)
        for name in TOOLS:
            self.populations[name] = NeuralPopulation(
                name=name, tau=0.8, threshold=0.35, baseline=0.1,
                refractory_period=0.2,
            )

        # Modes: very slow (momentum), high baseline
        for name in MODES:
            self.populations[name] = NeuralPopulation(
                name=name, tau=3.0, threshold=0.5, baseline=0.3,
                refractory_period=1.0,
            )

    def _init_connections(self):
        """Seed biologically-inspired connections.

        These are STARTING weights — they evolve through STDP.
        The magic: even if these initial weights are wrong,
        learning will correct them over sessions.
        """
        c = self._connect  # shorthand

        # ── Sensory → Signal connections ──
        # "More observations + more patterns + high confidence → convergence"
        c("obs_count", "convergence", 0.2)
        c("pat_count", "convergence", 0.3)
        c("prin_count", "convergence", 0.5)
        c("confidence_avg", "convergence", 0.6)
        c("round_num", "convergence", 0.2)

        # "Contradictions → contradiction signal"
        c("contra_count", "contradiction", 0.7)

        # "Few new patterns → diminishing returns"
        c("pat_count", "diminishing", -0.3, sign=-1)  # More patterns = less diminishing
        c("round_num", "diminishing", 0.4)

        # "Model failures → anomaly"
        c("failure_count", "anomaly", 0.6)

        # "Convergence + principles → sufficient depth"
        c("convergence", "sufficient", 0.6)
        c("prin_count", "sufficient", 0.3)
        c("round_num", "sufficient", 0.15)

        # ── Signal → Signal cross-connections ──
        # Convergence inhibits anomaly (if converging, less worry about anomalies)
        c("convergence", "anomaly", -0.3, sign=-1)
        # Anomaly inhibits convergence (anomalies mean we haven't converged)
        c("anomaly", "convergence", -0.3, sign=-1)
        # Contradiction excites anomaly
        c("contradiction", "anomaly", 0.3)
        # Diminishing excites sufficient
        c("diminishing", "sufficient", 0.2)

        # ── Signal → Mode connections ──
        # Anomaly/contradiction → explore mode
        c("anomaly", "explore", 0.5)
        c("contradiction", "explore", 0.4)
        c("obs_hunger", "explore", 0.3)  # Hungry = explore
        # Convergence/diminishing/confidence → integrate mode
        c("convergence", "integrate", 0.6)
        c("diminishing", "integrate", 0.3)
        c("confidence_avg", "integrate", 0.4)
        c("prin_count", "integrate", 0.3)
        # Modes inhibit each other (mutual inhibition = winner-take-all)
        c("explore", "integrate", -0.6, sign=-1)
        c("integrate", "explore", -0.6, sign=-1)

        # ── Mode → Tool connections ──
        # Explore mode boosts sensory/creative tools
        for tool in ["observe", "body_think", "shift_dimension", "play", "imagine", "empathize"]:
            c("explore", tool, 0.3)
        # Integrate mode boosts analytical tools
        for tool in ["abstract", "analogize", "model", "synthesize", "recognize_patterns"]:
            c("integrate", tool, 0.3)

        # ── Hunger → Tool connections (absence drives action) ──
        c("obs_hunger", "observe", 0.7)             # Starving for observations → OBSERVE
        c("obs_hunger", "body_think", 0.4)           # Also try body sensing
        c("obs_hunger", "shift_dimension", 0.3)      # Try different angles
        c("pat_hunger", "recognize_patterns", 0.5)   # Need patterns → find them
        c("pat_hunger", "form_patterns", 0.3)         # Try forming new ones
        c("prin_hunger", "abstract", 0.5)             # Need principles → abstract
        c("prin_hunger", "analogize", 0.3)            # Try analogy

        # ── Sensory → Tool connections ──
        # Observations exist → patterns should fire
        c("obs_count", "recognize_patterns", 0.4)
        # Patterns exist → abstract should fire
        c("pat_count", "abstract", 0.4)
        # Principles exist → analogize, model, imagine, play
        c("prin_count", "analogize", 0.3)
        c("prin_count", "model", 0.3)
        c("prin_count", "imagine", 0.25)
        c("prin_count", "play", 0.25)
        # Contradictions → empathize (understand conflict from inside)
        c("contra_count", "empathize", 0.3)
        # Failures → body_think, shift_dimension (try different approach)
        c("failure_count", "body_think", 0.3)
        c("failure_count", "shift_dimension", 0.3)

        # ── Tool → Tool sequential connections (pipeline) ──
        c("observe", "recognize_patterns", 0.3)
        c("recognize_patterns", "form_patterns", 0.25)
        c("form_patterns", "abstract", 0.3)
        c("abstract", "analogize", 0.3)
        c("analogize", "model", 0.25)
        c("model", "synthesize", 0.3)
        # Creative tools feed back to pattern recognition
        c("body_think", "recognize_patterns", 0.2)
        c("shift_dimension", "recognize_patterns", 0.2)
        c("empathize", "abstract", 0.15)
        c("play", "abstract", 0.15)
        c("imagine", "model", 0.2)
        c("transform", "synthesize", 0.2)

        # ── Lateral inhibition between tools ──
        # Tools that serve similar functions compete
        for a, b in [
            ("observe", "body_think"), ("observe", "shift_dimension"),
            ("recognize_patterns", "form_patterns"),
            ("abstract", "analogize"),
            ("imagine", "play"),
        ]:
            c(a, b, -0.15, sign=-1, plasticity=False)
            c(b, a, -0.15, sign=-1, plasticity=False)

        # Synthesize is always weakly excited (final integrator)
        c("prin_count", "synthesize", 0.2)
        c("confidence_avg", "synthesize", 0.15)

    def _connect(self, source: str, target: str, weight: float,
                 sign: int = 1, plasticity: bool = True):
        """Add a connection. sign=1 excitatory, sign=-1 inhibitory."""
        self.connections.append(Connection(
            source=source, target=target,
            weight=abs(weight), sign=sign,
            plasticity=plasticity,
        ))

    # ─── Main Update ───

    def update(self, sensory_input: dict[str, float], dt: float = 1.0):
        """One full time step of the circuit.

        1. Clamp sensory neurons to input values
        2. Compute input currents for all populations
        3. Update all populations (leaky integration)
        4. Apply STDP learning
        5. Homeostatic plasticity
        6. Update neuromodulators
        """
        self.time_step += 1

        # 1. Clamp sensory inputs
        for name, value in sensory_input.items():
            if name in self.populations:
                self.populations[name].rate = max(0.0, min(1.0, value))
                self.populations[name].fired = value > self.populations[name].threshold

        # 2. Compute input currents from connections
        currents: dict[str, float] = {name: 0.0 for name in self.populations}
        for conn in self.connections:
            if conn.source in self.populations:
                source_rate = self.populations[conn.source].rate
                currents[conn.target] = currents.get(conn.target, 0.0) + \
                    source_rate * conn.effective_weight

        # 3. Update all non-sensory populations
        # NE increases noise (stochastic resonance under novelty/arousal)
        # ACh decreases noise (improved signal fidelity under focused attention)
        ne = 0.5 if self.ablate_norepinephrine else self.norepinephrine
        ach = 0.5 if self.ablate_acetylcholine else self.acetylcholine
        effective_noise = 0.02 * (1.0 + 0.5 * ne) * (1.0 - 0.3 * ach)
        for name, pop in self.populations.items():
            if name not in sensory_input:  # Don't update clamped sensory neurons
                # Apply neuromodulatory gain
                pop.gain = self._compute_gain(name)
                pop.step(currents.get(name, 0.0), dt=dt, noise=effective_noise)

        # 4. STDP learning
        if not self.ablate_stdp:
            self._apply_stdp()

        # 5. Homeostatic plasticity (every 5 steps)
        if self.time_step % 5 == 0 and not self.ablate_homeostatic:
            self._homeostatic_plasticity()

        # 6. Update neuromodulators
        self._update_neuromodulators()

    def _compute_gain(self, pop_name: str) -> float:
        """Neuromodulatory gain control.

        NE increases gain globally (arousal).
        DA increases gain for recently-rewarded populations.
        ACh increases gain for learning-related populations.
        """
        ne = 0.5 if self.ablate_norepinephrine else self.norepinephrine
        da = 0.0 if self.ablate_dopamine else self.dopamine
        ach = 0.5 if self.ablate_acetylcholine else self.acetylcholine

        base_gain = 0.8 + 0.4 * ne  # 0.8 - 1.2

        if pop_name in TOOLS:
            # Dopamine boosts tool activation when reward is positive
            base_gain += 0.2 * max(0, da)

        if pop_name in SIGNALS:
            # ACh increases signal sensitivity (learning precision)
            base_gain += 0.15 * ach

        return max(0.3, min(2.0, base_gain))

    def _apply_stdp(self):
        """Spike-Timing Dependent Plasticity with Homeostatic Learning Rate.

        Kingson Man et al. insight: each tool's learning rate self-adjusts.
        - Good tools: lr decreases → stable, exploit what works
        - Bad tools: lr increases → rapid relearning, explore alternatives
        - Domain shift: all lr spike → fast unlearning + relearning

        Standard STDP:
        - Source fires BEFORE target → strengthen (LTP)
        - Source fires AFTER target → weaken (LTD)
        - Magnitude modulated by dopamine (reward-modulated STDP)
        """
        da = 0.0 if self.ablate_dopamine else self.dopamine
        ach = 0.5 if self.ablate_acetylcholine else self.acetylcholine
        base_lr = self.stdp_lr * (1.0 + abs(da) * ach)

        # Domain shift multiplier: if detected, 3x learning rate for rapid relearning
        shift_mult = 3.0 if self.domain_shift_detected else 1.0

        for conn in self.connections:
            if not conn.plasticity:
                continue

            source = self.populations.get(conn.source)
            target = self.populations.get(conn.target)
            if not source or not target:
                continue

            # Use per-tool adaptive lr if target is a tool population
            if conn.target in TOOLS and conn.target in self.tool_lr:
                lr = self.tool_lr[conn.target] * (1.0 + abs(da) * ach) * shift_mult
            else:
                lr = base_lr * shift_mult

            if source.fired and target.fired:
                # Both fired this step → strengthen (co-activation)
                dt_spike = source.last_fire_time - target.last_fire_time
                if abs(dt_spike) <= self.stdp_window:
                    # LTP: source fired before or with target
                    dw = lr * math.exp(-abs(dt_spike) / max(self.stdp_window, 1))
                    if da >= 0:
                        conn.weight = min(1.0, conn.weight + dw)
                    else:
                        # Negative dopamine → LTD instead
                        conn.weight = max(0.01, conn.weight - dw * 0.5)

            elif source.fired and not target.fired:
                # Source fired but target didn't → slight weakening
                conn.weight = max(0.01, conn.weight - lr * 0.1)

        # Enforce Dale's Law: weight represents magnitude, sign encodes polarity
        for conn in self.connections:
            if conn.weight < 0.0:
                conn.weight = 0.0

        # Record fire times
        for name, pop in self.populations.items():
            if pop.fired:
                pop.last_fire_time = self.time_step

    def _homeostatic_plasticity(self):
        """Keep average firing rates near target.

        If a population fires too much → reduce incoming weights.
        If too little → increase incoming weights.
        This prevents runaway excitation and silent neurons.
        """
        for name, pop in self.populations.items():
            if name in SENSORY:
                continue  # Don't adjust sensory neurons

            error = pop.rate - self.target_rate
            if abs(error) < 0.05:
                continue  # Close enough

            # Threshold homeostasis (Turrigiano 1999): slow adaptation (tau=100) prevents
            # populations from permanently saturating or going silent.
            pop.threshold = max(0.1, min(0.9, pop.threshold + error / 100.0))

            # Scale all incoming connections
            scale = 1.0 - 0.06 * error  # Over-active → shrink, under-active → grow
            for conn in self.connections:
                if conn.target == name and conn.plasticity:
                    conn.weight = max(0.01, min(1.0, conn.weight * scale))

    def _update_neuromodulators(self):
        """Update neuromodulators based on circuit state.

        Unlike if-else: modulators respond to NEURAL ACTIVITY, not state variables.
        Ablated modulators are clamped to neutral values.
        """
        # Dopamine: prediction error = (actual convergence - expected)
        if not self.ablate_dopamine:
            conv_rate = self.populations.get("convergence", NeuralPopulation(name="x")).rate
            self.dopamine = 0.8 * self.dopamine + 0.2 * (conv_rate - 0.5)
        else:
            self.dopamine = 0.0

        # Norepinephrine: high when anomaly/contradiction, low when converging
        if not self.ablate_norepinephrine:
            anomaly_rate = self.populations.get("anomaly", NeuralPopulation(name="x")).rate
            contra_rate = self.populations.get("contradiction", NeuralPopulation(name="x")).rate
            arousal = (anomaly_rate + contra_rate) / 2
            self.norepinephrine = 0.7 * self.norepinephrine + 0.3 * (0.3 + arousal)
        else:
            self.norepinephrine = 0.5

        # Acetylcholine: high early (learning), low late (exploitation)
        if not self.ablate_acetylcholine:
            round_rate = self.populations.get("round_num", NeuralPopulation(name="x")).rate
            self.acetylcholine = 0.8 * self.acetylcholine + 0.2 * (1.0 - round_rate * 0.5)
        else:
            self.acetylcholine = 0.5

    # ─── Read State ───

    def get_signal(self, name: str) -> float:
        """Get firing rate of a signal population."""
        pop = self.populations.get(name)
        return pop.rate if pop else 0.0

    def get_fired(self, name: str) -> bool:
        """Did this population fire this step?"""
        pop = self.populations.get(name)
        return pop.fired if pop else False

    def get_tool_activations(self) -> dict[str, float]:
        """Get activation level of each tool population."""
        return {name: self.populations[name].rate for name in TOOLS if name in self.populations}

    def get_active_tools(self, threshold: float = 0.3) -> list[str]:
        """Which tools are above activation threshold?"""
        activations = self.get_tool_activations()
        return sorted(
            [name for name, rate in activations.items() if rate >= threshold],
            key=lambda n: -activations[n],
        )

    def get_mode(self) -> str:
        """Current mode: explore, integrate, or balanced."""
        explore = self.get_signal("explore")
        integrate = self.get_signal("integrate")
        if explore > integrate + 0.1:
            return "sympathetic"
        elif integrate > explore + 0.1:
            return "parasympathetic"
        return "balanced"

    # ─── Encode State ───

    @staticmethod
    def encode_state(state) -> dict[str, float]:
        """Convert CognitiveState to sensory input values (0-1 normalized).

        Key biological insight: ABSENCE is a signal, not zero.
        Hunger = high glucose-absence signal, not low glucose signal.
        So we encode both presence AND absence as positive signals.
        """
        n_obs = len(state.observations)
        n_pat = len(state.patterns)
        n_prin = len(state.principles)
        n_contra = len(state.contradictions)
        n_failures = sum(len(m.failures) for m in state.model_results)
        avg_conf = (sum(p.confidence for p in state.principles) / max(n_prin, 1)) if n_prin else 0

        return {
            # Presence signals (high when data exists)
            "obs_count": min(1.0, n_obs / 50),
            "pat_count": min(1.0, n_pat / 20),
            "prin_count": min(1.0, n_prin / 10),
            "contra_count": min(1.0, n_contra / 5),
            "failure_count": min(1.0, n_failures / 10),
            "round_num": min(1.0, state.round / 5),
            "confidence_avg": avg_conf,
            "cost_ratio": getattr(state.signals, 'total_cost', 0) / 20.0,
            # Absence signals (high when data is MISSING — like hunger)
            # Denominators match presence signals for consistency
            "obs_hunger": max(0.0, 1.0 - n_obs / 50),    # Starving for observations
            "pat_hunger": max(0.0, 1.0 - n_pat / 15),     # Need patterns
            "prin_hunger": max(0.0, 1.0 - n_prin / 8),    # Need principles
        }

    # ─── Record Tool Outcome (for STDP) ───

    def record_tool_outcome(self, tool_name: str, success: bool):
        """After a tool runs, modulate dopamine and adaptive learning rate.

        Kingson Man "skin in the game": the tool's own performance directly
        affects its ability to learn. Good → stable (low lr). Bad → explore (high lr).
        """
        if not self.ablate_dopamine:
            if success:
                self.dopamine = min(1.0, self.dopamine + 0.3)
            else:
                self.dopamine = max(-1.0, self.dopamine - 0.2)

        # Also boost/suppress the tool's population
        if tool_name in self.populations:
            if success:
                self.populations[tool_name].baseline = min(0.3,
                    self.populations[tool_name].baseline + 0.02)
            else:
                self.populations[tool_name].baseline = max(0.0,
                    self.populations[tool_name].baseline - 0.01)

        # ── Homeostatic Learning Rate (Kingson Man) ──
        # Update exponential moving average of performance
        alpha = 0.3  # EMA decay
        prev_perf = self.tool_performance.get(tool_name, 0.5)
        new_perf = alpha * (1.0 if success else 0.0) + (1 - alpha) * prev_perf
        self.tool_performance[tool_name] = new_perf

        # Adapt learning rate: bad performance → high lr (explore), good → low lr (exploit)
        # Inverted: lr = lr_max - (lr_max - lr_min) * performance
        new_lr = self.lr_max - (self.lr_max - self.lr_min) * new_perf
        self.tool_lr[tool_name] = max(self.lr_min, min(self.lr_max, new_lr))

    # ─── Persistence ───

    @staticmethod
    def _circuit_dir() -> Path:
        return Path.home() / ".sparks" / "circuits"

    def _to_dict(self) -> dict:
        """Serialize circuit state to dict."""
        return {
            "connections": [
                {"source": c.source, "target": c.target,
                 "weight": c.weight, "sign": c.sign,
                 "plasticity": c.plasticity}
                for c in self.connections
            ],
            "baselines": {
                name: pop.baseline
                for name, pop in self.populations.items()
            },
            "modulators": {
                "dopamine": self.dopamine,
                "norepinephrine": self.norepinephrine,
                "acetylcholine": self.acetylcholine,
            },
            "homeostatic_lr": {
                "tool_lr": self.tool_lr,
                "tool_performance": self.tool_performance,
            },
            "time_step": self.time_step,
        }

    def _from_dict(self, data: dict) -> bool:
        """Restore circuit state from dict."""
        try:
            saved_conns = {(c["source"], c["target"]): c for c in data.get("connections", [])}
            for conn in self.connections:
                key = (conn.source, conn.target)
                if key in saved_conns:
                    conn.weight = saved_conns[key]["weight"]
                    conn.plasticity = saved_conns[key].get("plasticity", True)

            for name, baseline in data.get("baselines", {}).items():
                if name in self.populations:
                    self.populations[name].baseline = baseline

            mods = data.get("modulators", {})
            self.dopamine = mods.get("dopamine", 0.0)
            self.norepinephrine = mods.get("norepinephrine", 0.5)
            self.acetylcholine = mods.get("acetylcholine", 0.5)

            # Restore homeostatic lr
            hlr = data.get("homeostatic_lr", {})
            self.tool_lr = hlr.get("tool_lr", {})
            self.tool_performance = hlr.get("tool_performance", {})

            self.time_step = data.get("time_step", 0)
            return True
        except Exception:
            return False

    def save(self, path: Optional[str] = None, domain: Optional[str] = None):
        """Save circuit state. If domain given, save domain-specific snapshot too."""
        # Save to default path (backward compat)
        save_path = Path(path) if path else Path.home() / ".sparks" / "circuit.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        save_path.write_text(json.dumps(data, indent=2))

        # Save domain-specific snapshot
        if domain:
            domain_dir = self._circuit_dir()
            domain_dir.mkdir(parents=True, exist_ok=True)
            domain_path = domain_dir / f"{_sanitize(domain)}.json"
            # Keep backup of previous snapshot for rollback
            if domain_path.exists():
                backup_path = domain_dir / f"{_sanitize(domain)}.backup.json"
                backup_path.write_text(domain_path.read_text())
            domain_path.write_text(json.dumps(data, indent=2))

    def load(self, path: Optional[str] = None, domain: Optional[str] = None):
        """Load circuit state. Prefers domain-specific if available."""
        # Try domain-specific first
        if domain:
            domain_path = self._circuit_dir() / f"{_sanitize(domain)}.json"
            if domain_path.exists():
                try:
                    data = json.loads(domain_path.read_text())
                    return self._from_dict(data)
                except Exception:
                    pass

        # Fall back to default
        load_path = Path(path) if path else Path.home() / ".sparks" / "circuit.json"
        if not load_path.exists():
            return False
        try:
            data = json.loads(load_path.read_text())
            return self._from_dict(data)
        except Exception:
            return False

    def rollback(self, domain: str) -> bool:
        """Rollback to previous domain snapshot."""
        backup_path = self._circuit_dir() / f"{_sanitize(domain)}.backup.json"
        if not backup_path.exists():
            return False
        try:
            data = json.loads(backup_path.read_text())
            self._from_dict(data)
            # Overwrite current with backup
            domain_path = self._circuit_dir() / f"{_sanitize(domain)}.json"
            domain_path.write_text(backup_path.read_text())
            return True
        except Exception:
            return False

    def detect_drift(self, domain: Optional[str] = None) -> dict:
        """Detect weight drift from initial values.

        Returns dict with drift statistics. High drift may indicate
        contamination from a different domain.
        """
        # Compare current weights against a fresh circuit
        fresh = NeuralCircuit()
        fresh_conns = {(c.source, c.target): c.weight for c in fresh.connections}

        drifts = []
        for conn in self.connections:
            key = (conn.source, conn.target)
            if key in fresh_conns:
                drifts.append(abs(conn.weight - fresh_conns[key]))

        if not drifts:
            return {"mean_drift": 0, "max_drift": 0, "anomalous": False}

        mean_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        # Anomaly: mean drift > 0.3 or any single connection drifted > 0.7
        anomalous = mean_drift > 0.3 or max_drift > 0.7

        return {
            "mean_drift": round(mean_drift, 4),
            "max_drift": round(max_drift, 4),
            "n_connections": len(drifts),
            "anomalous": anomalous,
            "domain": domain,
        }

    def reset(self):
        """Reset all weights and baselines to initial values."""
        self.connections = []
        self._init_connections()
        for name in self.populations:
            self.populations[name].baseline = 0.1 if name in TOOLS else 0.0
            self.populations[name].rate = 0.0
            self.populations[name].refractory = 0.0
        self.dopamine = 0.0
        self.norepinephrine = 0.5
        self.acetylcholine = 0.5
        self.time_step = 0

    # ─── Debug ───

    def status(self) -> str:
        """Human-readable circuit status."""
        lines = [f"═══ Circuit (t={self.time_step}) ═══"]
        lines.append(f"Mode: {self.get_mode()} | DA={self.dopamine:.2f} NE={self.norepinephrine:.2f} ACh={self.acetylcholine:.2f}")

        lines.append("\nSignals:")
        for name in SIGNALS:
            pop = self.populations.get(name)
            if pop:
                icon = "●" if pop.fired else "○"
                lines.append(f"  {icon} {name}: {pop.rate:.2f} (thr={pop.threshold:.2f})")

        lines.append("\nTools (top 8):")
        activations = self.get_tool_activations()
        for name in sorted(activations, key=activations.get, reverse=True)[:8]:
            pop = self.populations[name]
            icon = "▶" if pop.fired else "·"
            lines.append(f"  {icon} {name}: {activations[name]:.2f}")

        return "\n".join(lines)


def _sanitize(name: str) -> str:
    """Sanitize domain name for filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower()).strip("_")[:50]
