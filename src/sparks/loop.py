"""Full Loop — B→F: Validate, Evolve, Apply, Feedback, Repeat.

Phase A = engine.run() (extract principles from data)
Phase B = validate (test principles against new data)
Phase C = evolve (strengthen/weaken/mutate principles)
Phase D = apply (generate predictions from principles)
Phase E = feedback (compare predictions to outcomes)
Phase F = auto-loop (B→E on each new data batch)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from sparks.cost import CostTracker, DEPTH_BUDGETS, DepthBudget
from sparks.data import DataStore
from sparks.llm import llm_structured, llm_call
from sparks.state import Principle

console = Console()

LOOP_HOME = Path.home() / ".sparks" / "loop"


# ─── Schemas ───


class ValidationResult(BaseModel):
    principle: str
    supported: bool
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    accuracy: float = 0.5  # 0-1
    needs_refinement: str = ""  # empty = fine, otherwise = suggestion


class ValidationBatch(BaseModel):
    results: list[ValidationResult]


class Prediction(BaseModel):
    principle_used: str
    prediction: str
    confidence: float = 0.5
    observable_by: str = ""  # what would confirm/deny this
    timeframe: str = ""


class PredictionBatch(BaseModel):
    predictions: list[Prediction]


class FeedbackResult(BaseModel):
    prediction: str
    outcome: str
    correct: bool
    explanation: str = ""
    principle_adjustment: str = ""  # "strengthen" | "weaken" | "refine" | "drop"


class FeedbackBatch(BaseModel):
    results: list[FeedbackResult]


class EvolvedPrinciple(BaseModel):
    statement: str
    confidence: float
    action: str  # "keep" | "refine" | "drop"
    reason: str = ""


class EvolutionBatch(BaseModel):
    principles: list[EvolvedPrinciple]


# ─── Persistent Principle Store ───


class PrincipleStore:
    """Principles that persist and evolve across loop cycles."""

    def __init__(self, name: str = "default"):
        self.path = LOOP_HOME / f"{name}_principles.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.principles: list[dict] = []
        self.history: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.principles = data.get("principles", [])
                self.history = data.get("history", [])
            except Exception:
                self.principles = []
                self.history = []

    def save(self):
        self.path.write_text(json.dumps({
            "principles": self.principles,
            "history": self.history,
        }, indent=2, ensure_ascii=False))

    def load_from_output(self, output_path: str):
        """Load principles from a Sparks analysis output (.md file)."""
        text = Path(output_path).read_text()
        principles = []
        current = None

        for line in text.split("\n"):
            if line.startswith("### Principle"):
                if current:
                    principles.append(current)
                # Extract statement after the "### Principle N: " prefix
                parts = line.split(": ", 1)
                statement = parts[1] if len(parts) > 1 else line
                current = {
                    "id": f"prin_{uuid.uuid4().hex[:8]}",
                    "statement": statement,
                    "confidence": 0.7,
                    "validations": 0,
                    "successes": 0,
                    "created": datetime.now().isoformat(),
                }
            elif line.startswith("**Confidence**:") and current:
                try:
                    conf_str = line.split(":")[1].strip().rstrip("%")
                    current["confidence"] = float(conf_str) / 100
                except (ValueError, IndexError):
                    pass

        if current:
            principles.append(current)

        self.principles = principles
        self.save()
        return len(principles)

    def update_from_validation(self, results: list[ValidationResult]):
        """Update principles based on validation results."""
        for vr in results:
            for p in self.principles:
                if _match(p["statement"], vr.principle):
                    p["validations"] = p.get("validations", 0) + 1
                    if vr.supported:
                        p["successes"] = p.get("successes", 0) + 1
                    # EMA confidence update
                    lr = 0.3
                    p["confidence"] = (1 - lr) * p["confidence"] + lr * vr.accuracy
                    break
        self.save()

    def update_from_evolution(self, evolved: list[EvolvedPrinciple]):
        """Apply evolution results — refine, keep, or drop principles."""
        new_principles = []
        for ep in evolved:
            if ep.action == "drop":
                # Log drop to history
                for p in self.principles:
                    if _match(p["statement"], ep.statement):
                        self.history.append({
                            **p,
                            "dropped": datetime.now().isoformat(),
                            "reason": ep.reason,
                        })
                        break
                continue

            # Find matching principle
            found = False
            for p in self.principles:
                if _match(p["statement"], ep.statement):
                    if ep.action == "refine":
                        p["statement"] = ep.statement
                        p["confidence"] = ep.confidence
                    new_principles.append(p)
                    found = True
                    break

            if not found and ep.action in ("keep", "refine"):
                # New principle from evolution
                new_principles.append({
                    "id": f"evo_{uuid.uuid4().hex[:8]}",
                    "statement": ep.statement,
                    "confidence": ep.confidence,
                    "validations": 0,
                    "successes": 0,
                    "created": datetime.now().isoformat(),
                })

        self.principles = new_principles
        self.save()

    def summary(self) -> str:
        lines = []
        for i, p in enumerate(self.principles, 1):
            v = p.get("validations", 0)
            s = p.get("successes", 0)
            rate = f"{s}/{v}" if v > 0 else "untested"
            lines.append(f"{i}. [{p['confidence']:.0%}] ({rate}) {p['statement'][:80]}...")
        return "\n".join(lines)


# ─── Phase B: Validate ───


def validate(
    principles: PrincipleStore,
    new_data: DataStore,
    tracker: CostTracker,
) -> list[ValidationResult]:
    """Test each principle against new data. Does the data support it?"""

    principles_text = "\n".join(
        f"{i+1}. [{p['confidence']:.0%}] {p['statement']}"
        for i, p in enumerate(principles.principles)
    )

    data_text = new_data.all_text(max_chars=80000)

    prompt = f"""You are a PRINCIPLE VALIDATOR. Test each principle against new data.

## Principles to Validate
{principles_text}

## New Data
{data_text}

## Instructions
For EACH principle:
1. Search the new data for evidence that SUPPORTS it
2. Search for evidence that CONTRADICTS it
3. Determine: is this principle supported by the new data?

Be RUTHLESS. A principle must earn its confidence.

For each result:
- principle: the principle statement (abbreviated)
- supported: true/false — does the new data support it?
- evidence_for: list of supporting evidence from the data
- evidence_against: list of contradicting evidence
- accuracy: 0.0-1.0 how well this principle explains the new data
- needs_refinement: empty string if fine, otherwise suggest how to refine"""

    result = llm_structured(
        prompt,
        model=tracker.select_model("validate"),
        schema=ValidationBatch,
        tool="validate",
        tracker=tracker,
        max_tokens=8192,
    )

    principles.update_from_validation(result.results)
    return result.results


# ─── Phase C: Evolve ───


def evolve(
    principles: PrincipleStore,
    validation_results: list[ValidationResult],
    tracker: CostTracker,
) -> list[EvolvedPrinciple]:
    """Strengthen/weaken/refine/drop principles based on validation."""

    principles_text = "\n".join(
        f"{i+1}. [{p['confidence']:.0%}] (tested {p.get('validations',0)}x, "
        f"passed {p.get('successes',0)}x) {p['statement']}"
        for i, p in enumerate(principles.principles)
    )

    validation_text = "\n".join(
        f"- {'SUPPORTED' if v.supported else 'FAILED'} [{v.accuracy:.0%}]: "
        f"{v.principle[:60]}... "
        f"{'Refine: ' + v.needs_refinement if v.needs_refinement else ''}"
        for v in validation_results
    )

    prompt = f"""You are a PRINCIPLE EVOLVER. Based on validation results, evolve the principles.

## Current Principles (with track record)
{principles_text}

## Latest Validation Results
{validation_text}

## Instructions
For each principle, decide:
- **keep**: Validation supports it. No change needed.
- **refine**: Partially supported but needs adjustment. Rewrite the statement
  to be more accurate/general. Keep the core insight, fix the edges.
- **drop**: Repeatedly failed validation. Not a real principle.

Also: if validation revealed a NEW pattern not captured by existing principles,
add it as a new principle with action="refine".

Rules:
- Don't drop a principle after just 1 failure — could be domain mismatch
- Refine > drop. Principles represent hard-won insight.
- New principles from validation should be specific and testable

For each principle:
- statement: the (possibly refined) principle statement
- confidence: updated confidence 0.0-1.0
- action: "keep" | "refine" | "drop"
- reason: why this decision"""

    result = llm_structured(
        prompt,
        model=tracker.select_model("evolve"),
        schema=EvolutionBatch,
        tool="evolve",
        tracker=tracker,
        max_tokens=8192,
    )

    principles.update_from_evolution(result.principles)
    return result.principles


# ─── Phase D: Apply (Predict) ───


def predict(
    principles: PrincipleStore,
    new_input: str,
    tracker: CostTracker,
) -> list[Prediction]:
    """Generate predictions from validated principles for a new situation."""

    principles_text = "\n".join(
        f"- [{p['confidence']:.0%}] {p['statement']}"
        for p in principles.principles
    )

    prompt = f"""You are a PREDICTION ENGINE. Apply validated principles to a new situation.

## Validated Principles
{principles_text}

## New Situation / Input
{new_input}

## Instructions
For each applicable principle, generate a SPECIFIC, TESTABLE prediction:

1. Which principle applies to this situation?
2. What does it predict will happen?
3. How confident are you? (based on principle confidence + fit to situation)
4. What would we observe if the prediction is correct?
5. By when should this be observable?

Rules:
- Only generate predictions from principles that CLEARLY apply
- Each prediction must be falsifiable — vague predictions are useless
- Confidence should be LOWER than the principle's confidence
  (principle = general law, prediction = specific application)

For each prediction:
- principle_used: which principle
- prediction: specific testable prediction
- confidence: 0.0-1.0
- observable_by: what evidence would confirm/deny
- timeframe: when this should be visible"""

    result = llm_structured(
        prompt,
        model=tracker.select_model("predict"),
        schema=PredictionBatch,
        tool="predict",
        tracker=tracker,
    )

    return result.predictions


# ─── Phase E: Feedback ───


def feedback(
    predictions: list[Prediction],
    outcomes: str,
    principles: PrincipleStore,
    tracker: CostTracker,
) -> list[FeedbackResult]:
    """Compare predictions to actual outcomes. Update principles."""

    pred_text = "\n".join(
        f"{i+1}. [{p.confidence:.0%}] {p.prediction}\n"
        f"   Based on: {p.principle_used[:60]}...\n"
        f"   Observable by: {p.observable_by}"
        for i, p in enumerate(predictions)
    )

    prompt = f"""You are a FEEDBACK JUDGE. Compare predictions to actual outcomes.

## Predictions Made
{pred_text}

## Actual Outcomes
{outcomes}

## Instructions
For each prediction:
1. Was it correct? (fully, partially, or wrong)
2. Why was it right/wrong?
3. What should happen to the underlying principle?
   - "strengthen" if prediction was correct
   - "weaken" if prediction was wrong but principle might still be valid
   - "refine" if partially correct — the principle needs adjustment
   - "drop" if prediction was completely wrong AND principle seems flawed

For each result:
- prediction: the prediction (abbreviated)
- outcome: what actually happened
- correct: true/false
- explanation: why it was right/wrong
- principle_adjustment: "strengthen" | "weaken" | "refine" | "drop" """

    result = llm_structured(
        prompt,
        model=tracker.select_model("feedback"),
        schema=FeedbackBatch,
        tool="feedback",
        tracker=tracker,
    )

    # Apply feedback to principle confidence
    for fr in result.results:
        for p in principles.principles:
            if _match(p["statement"], fr.prediction) or _match(p["statement"], fr.prediction):
                lr = 0.2
                if fr.principle_adjustment == "strengthen":
                    p["confidence"] = min(0.99, p["confidence"] + lr * (1.0 - p["confidence"]))
                elif fr.principle_adjustment == "weaken":
                    p["confidence"] = max(0.1, p["confidence"] - lr * 0.3)
                elif fr.principle_adjustment == "drop":
                    p["confidence"] = max(0.05, p["confidence"] * 0.5)
                break

    principles.save()
    return result.results


# ─── Phase F: Full Loop ───


def run_loop(
    principles_source: str,
    data_dirs: list[str],
    cycles: int = 1,
    budget: float = 10.0,
    predict_input: str = "",
    outcomes: str = "",
) -> PrincipleStore:
    """Run the full B→E loop.

    Args:
        principles_source: path to .md output or existing store name
        data_dirs: list of new data directories for validation
        cycles: how many B→C cycles to run
        budget: max cost
        predict_input: optional situation to predict
        outcomes: optional actual outcomes for feedback
    """
    tracker = CostTracker(DepthBudget(max_cost=budget, max_rounds=cycles, max_priority=3))

    # Load principles
    store = PrincipleStore("loop")
    if principles_source.endswith(".md"):
        n = store.load_from_output(principles_source)
        console.print(f"[bold]Loaded {n} principles from {principles_source}[/]")
    else:
        store = PrincipleStore(principles_source)
        console.print(f"[bold]Loaded {len(store.principles)} existing principles[/]")

    console.print(f"\n{store.summary()}\n")

    # ── Phase B→C: Validate + Evolve for each data batch ──
    for cycle_num in range(cycles):
        console.print(f"\n[bold]═══ Cycle {cycle_num + 1}/{cycles} ═══[/]")

        for data_dir in data_dirs:
            if not tracker.can_afford(tracker.select_model("validate")):
                console.print("[yellow]Budget exhausted[/]")
                break

            console.print(f"\n[bold]Phase B: Validating against {data_dir}[/]")
            data = DataStore(data_dir)
            val_results = validate(store, data, tracker)

            # Report
            supported = sum(1 for v in val_results if v.supported)
            console.print(f"  Supported: {supported}/{len(val_results)}")
            for v in val_results:
                icon = "[green]✓[/]" if v.supported else "[red]✗[/]"
                console.print(f"  {icon} [{v.accuracy:.0%}] {v.principle[:60]}...")
                if v.needs_refinement:
                    console.print(f"    [yellow]→ {v.needs_refinement}[/]")

            # Phase C: Evolve
            if not tracker.can_afford(tracker.select_model("evolve")):
                break

            console.print(f"\n[bold]Phase C: Evolving principles[/]")
            evo_results = evolve(store, val_results, tracker)

            for ep in evo_results:
                if ep.action == "keep":
                    console.print(f"  [green]keep[/] [{ep.confidence:.0%}] {ep.statement[:60]}...")
                elif ep.action == "refine":
                    console.print(f"  [yellow]refine[/] [{ep.confidence:.0%}] {ep.statement[:60]}...")
                elif ep.action == "drop":
                    console.print(f"  [red]drop[/] {ep.statement[:60]}... — {ep.reason}")

        console.print(f"\n[dim]After cycle {cycle_num + 1}: {len(store.principles)} principles[/]")

    # ── Phase D: Predict (optional) ──
    predictions = []
    if predict_input and tracker.can_afford(tracker.select_model("predict")):
        console.print(f"\n[bold]Phase D: Generating predictions[/]")
        predictions = predict(store, predict_input, tracker)

        for pred in predictions:
            console.print(f"  [{pred.confidence:.0%}] {pred.prediction[:80]}...")
            console.print(f"    [dim]Based on: {pred.principle_used[:50]}...[/]")

    # ── Phase E: Feedback (optional) ──
    if outcomes and predictions and tracker.can_afford(tracker.select_model("feedback")):
        console.print(f"\n[bold]Phase E: Processing feedback[/]")
        fb_results = feedback(predictions, outcomes, store, tracker)

        for fr in fb_results:
            icon = "[green]✓[/]" if fr.correct else "[red]✗[/]"
            console.print(f"  {icon} {fr.prediction[:50]}...")
            console.print(f"    → {fr.principle_adjustment}: {fr.explanation[:60]}...")

    # ── Final Report ──
    console.print(f"\n[bold]═══ Final State ═══[/]")
    console.print(f"Principles: {len(store.principles)}")
    console.print(f"Cost: ${tracker.total_cost:.2f}")
    console.print(f"\n{store.summary()}")

    # Save final state
    store.save()
    console.print(f"\n[dim]Saved to {store.path}[/]")

    return store


# ─── Helpers ───


def _match(a: str, b: str) -> bool:
    """Quick fuzzy match between two principle statements."""
    a_words = set(a.lower().split()[:10])
    b_words = set(b.lower().split()[:10])
    if not a_words or not b_words:
        return False
    return len(a_words & b_words) / min(len(a_words), len(b_words)) > 0.4
