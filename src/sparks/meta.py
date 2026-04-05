"""Meta-Cognition Loop — Sparks improves itself by analyzing its own code.

    ┌──────────────────────────────────────────────────────────┐
    │                  META-COGNITION LOOP                      │
    │                                                          │
    │  ┌──────────┐     ┌───────────┐     ┌──────────────┐    │
    │  │ Analyze  │────►│ Generate  │────►│  Benchmark   │    │
    │  │ own code │     │ patches   │     │  before/after│    │
    │  └──────────┘     └───────────┘     └──────┬───────┘    │
    │       ▲                                     │            │
    │       │            ┌───────────┐            │            │
    │       └────────────│  Apply or │◄───────────┘            │
    │                    │  Rollback │                          │
    │                    └───────────┘                          │
    └──────────────────────────────────────────────────────────┘

The framework feeds its own source code as data, discovers architectural
principles and weaknesses, generates improvements, validates them via
benchmark, and applies only safe changes.

Based on Kingson Man et al., "Need is All You Need" (2022):
- The system has "skin in the game" — its own code quality directly
  affects its performance
- Self-regulated learning: weak components adapt faster
- Domain shift detection triggers rapid relearning
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from sparks.cost import CostTracker, DEPTH_BUDGETS
from sparks.llm import llm_call, llm_structured

console = Console()

META_HOME = Path.home() / ".sparks" / "meta"
SRC_DIR = Path(__file__).parent  # src/sparks/


# ─── Schemas ───


class ArchitecturalInsight(BaseModel):
    component: str           # Which file/module
    insight: str             # What was discovered
    severity: str            # "critical" | "improvement" | "minor"
    suggested_change: str    # Concrete suggestion


class CodePatch(BaseModel):
    file: str                # Relative to src/sparks/
    original: str            # Exact text to replace
    replacement: str         # New text
    reason: str
    risk: str = "low"        # "low" | "medium" | "high"


class MetaAnalysis(BaseModel):
    insights: list[ArchitecturalInsight]
    bottleneck: str
    top_improvement: str


class MetaPatchBatch(BaseModel):
    patches: list[CodePatch]


class BenchmarkResult(BaseModel):
    n_principles: int = 0
    confidence: float = 0.0
    cost: float = 0.0
    time_seconds: float = 0.0
    tools_used: list[str] = []
    error: str = ""


# ─── Phase 1: Self-Analysis ───


def analyze_own_code(
    focus: str = "all",
    tracker: Optional[CostTracker] = None,
) -> MetaAnalysis:
    """Run Sparks' 13 thinking tools on its own source code.

    Instead of analyzing external data, we feed src/sparks/*.py as input.
    The goal: find architectural weaknesses and improvement opportunities.
    """
    if not tracker:
        tracker = CostTracker(DEPTH_BUDGETS["standard"])

    # Collect source code
    code_parts = []
    focus_files = {
        "circuit": ["circuit.py"],
        "tools": list((SRC_DIR / "tools").glob("*.py")),
        "engine": ["autonomic.py", "engine.py", "nervous.py"],
        "all": None,  # everything
    }

    if focus == "all":
        for py_file in sorted(SRC_DIR.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text()
            if len(content) > 100:
                code_parts.append(f"### {py_file.name}\n```python\n{content[:4000]}\n```\n")
        for py_file in sorted((SRC_DIR / "tools").glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text()
            if len(content) > 100:
                code_parts.append(f"### tools/{py_file.name}\n```python\n{content[:3000]}\n```\n")
    else:
        targets = focus_files.get(focus, [])
        for target in targets:
            path = target if isinstance(target, Path) else SRC_DIR / target
            if path.exists():
                content = path.read_text()
                code_parts.append(f"### {path.name}\n```python\n{content[:5000]}\n```\n")

    code_text = "\n".join(code_parts)
    # Truncate to avoid token explosion
    if len(code_text) > 60000:
        code_text = code_text[:60000] + "\n[... truncated]"

    prompt = f"""You are a SYSTEMS ARCHITECT analyzing the source code of an AI cognitive framework called "Sparks."

## Context
Sparks has:
- 13 thinking tools (observe, imagine, abstract, patterns, analogize, body_think, empathize, shift_dimension, model, play, transform, synthesize)
- A neural circuit (~30 LIF populations, ~80 STDP connections) that drives tool execution order
- Neuromodulators (dopamine, norepinephrine, acetylcholine)
- Homeostatic learning rate (per-tool adaptive lr based on Kingson Man et al.)
- Self-optimization loop (analyzes output quality, fixes prompts)
- Full loop (validate → evolve → predict → feedback)

## Source Code
{code_text}

## Your Task
Analyze this code as if you are a senior architect reviewing for:

1. **Architectural weaknesses**: Design flaws, tight coupling, missing error handling at system boundaries
2. **Neural circuit improvements**: Are the connection weights well-calibrated? Missing connections? Redundant populations?
3. **Tool interaction issues**: Do tools duplicate work? Missing handoffs?
4. **Performance bottlenecks**: Unnecessary LLM calls, token waste, slow paths
5. **Biological plausibility gaps**: Where does the code diverge from neuroscience in ways that hurt performance?

For each insight:
- component: which file/module
- insight: what you found
- severity: critical / improvement / minor
- suggested_change: concrete, specific change

Also identify:
- bottleneck: the single biggest weakness
- top_improvement: the single change with highest impact"""

    result = llm_structured(
        prompt,
        model=tracker.select_model("optimize"),
        schema=MetaAnalysis,
        tool="meta_analyze",
        tracker=tracker,
        max_tokens=8192,
    )

    return result


# ─── Phase 2: Generate Patches ───


def generate_patches(
    analysis: MetaAnalysis,
    tracker: Optional[CostTracker] = None,
) -> MetaPatchBatch:
    """Convert architectural insights into concrete code patches."""
    if not tracker:
        tracker = CostTracker(DEPTH_BUDGETS["standard"])

    # Focus on critical + improvement severity only
    actionable = [i for i in analysis.insights if i.severity in ("critical", "improvement")]
    if not actionable:
        return MetaPatchBatch(patches=[])

    # Read the relevant files
    files_needed = set(i.component for i in actionable)
    file_contents = {}
    for fname in files_needed:
        path = SRC_DIR / fname
        if not path.exists():
            # Try tools/
            path = SRC_DIR / "tools" / fname
        if path.exists():
            file_contents[fname] = path.read_text()[:6000]

    insights_text = "\n".join(
        f"- [{i.severity}] {i.component}: {i.insight}\n  Suggestion: {i.suggested_change}"
        for i in actionable[:8]  # Limit to top 8
    )

    files_text = "\n".join(
        f"### {fname}\n```python\n{content}\n```"
        for fname, content in file_contents.items()
    )

    prompt = f"""You are a CODE SURGEON. Generate precise code patches from these architectural insights.

## Insights to Address
{insights_text}

## Current Code
{files_text}

## Rules
1. Each patch must have the EXACT original text that exists in the file
2. Replacement text must be syntactically valid Python
3. Changes should be MINIMAL — fix the issue, nothing else
4. Mark risk level: "low" (safe refactor), "medium" (behavior change), "high" (structural change)
5. Only generate patches for insights with clear, concrete fixes
6. Do NOT change function signatures that are called from other modules unless necessary

Generate a list of CodePatch objects."""

    result = llm_structured(
        prompt,
        model=tracker.select_model("optimize"),
        schema=MetaPatchBatch,
        tool="meta_patch",
        tracker=tracker,
        max_tokens=8192,
    )

    return result


# ─── Phase 3: Benchmark Before/After ───


def run_benchmark(
    data_path: str,
    depth: str = "quick",
) -> BenchmarkResult:
    """Quick benchmark: run Sparks and measure output quality."""
    try:
        from sparks.research import set_seed
        from sparks.autonomic import run_autonomic

        set_seed(42)
        start = time.time()
        result = run_autonomic(
            goal="Extract the core principles from this data",
            data_path=data_path,
            depth=depth,
        )
        elapsed = time.time() - start

        return BenchmarkResult(
            n_principles=len(result.principles),
            confidence=result.confidence,
            cost=result.total_cost,
            time_seconds=elapsed,
            tools_used=result.tools_used,
        )
    except Exception as e:
        return BenchmarkResult(error=str(e))


# ─── Phase 4: Apply with Rollback ───


def apply_patches(
    patches: MetaPatchBatch,
    max_risk: str = "medium",
) -> tuple[list[str], list[Path]]:
    """Apply patches, return (applied_log, backup_paths).

    Only applies patches up to max_risk level.
    Creates backups for rollback.
    """
    risk_levels = {"low": 0, "medium": 1, "high": 2}
    max_risk_level = risk_levels.get(max_risk, 1)

    backup_dir = META_HOME / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    applied = []
    backup_paths = []

    for patch in patches.patches:
        if risk_levels.get(patch.risk, 2) > max_risk_level:
            applied.append(f"SKIP [{patch.risk}] {patch.file}: {patch.reason}")
            continue

        # Find file
        path = SRC_DIR / patch.file
        if not path.exists():
            path = SRC_DIR / "tools" / patch.file
        if not path.exists():
            applied.append(f"SKIP {patch.file}: file not found")
            continue

        content = path.read_text()
        if patch.original not in content:
            applied.append(f"SKIP {patch.file}: original text not found")
            continue

        # Backup
        backup_path = backup_dir / patch.file.replace("/", "_")
        shutil.copy2(path, backup_path)
        backup_paths.append(backup_path)

        # Apply
        new_content = content.replace(patch.original, patch.replacement, 1)

        # Syntax check before writing
        try:
            compile(new_content, str(path), "exec")
        except SyntaxError as e:
            applied.append(f"REJECT {patch.file}: syntax error after patch — {e}")
            continue

        path.write_text(new_content)
        applied.append(f"APPLIED [{patch.risk}] {patch.file}: {patch.reason}")

    return applied, backup_paths


def rollback(backup_paths: list[Path]):
    """Restore files from backups."""
    for backup_path in backup_paths:
        # Reconstruct original path
        fname = backup_path.name.replace("_", "/", backup_path.name.count("_") - 1)
        # Simple: just check both locations
        for candidate in [SRC_DIR / fname, SRC_DIR / backup_path.name]:
            if candidate.exists():
                shutil.copy2(backup_path, candidate)
                break


# ─── Full Meta-Cognition Loop ───


def meta_loop(
    benchmark_data: str,
    max_iterations: int = 3,
    depth: str = "quick",
    apply: bool = False,
    max_risk: str = "low",
) -> dict:
    """Full self-improvement loop.

    For each iteration:
    1. Analyze own source code
    2. Generate code patches
    3. Benchmark BEFORE
    4. Apply patches
    5. Benchmark AFTER
    6. Keep if improved, rollback if degraded
    7. Repeat

    Args:
        benchmark_data: Path to data for benchmark runs
        max_iterations: Maximum improvement cycles
        depth: Benchmark depth ("quick" recommended for speed)
        apply: Actually apply changes (False = dry run)
        max_risk: Maximum risk level to apply ("low", "medium", "high")
    """
    tracker = CostTracker(DEPTH_BUDGETS["deep"])
    history = []

    console.print(f"\n[bold cyan]⚡ Meta-Cognition Loop[/]")
    console.print(f"Iterations: {max_iterations} | Benchmark depth: {depth} | Max risk: {max_risk}")
    console.print(f"Mode: {'LIVE' if apply else 'DRY RUN'}")

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Iteration {iteration}/{max_iterations}[/]")
        console.print(f"{'='*60}")

        # ── Phase 1: Analyze ──
        console.print(f"\n[bold]Phase 1: Analyzing own source code...[/]")
        analysis = analyze_own_code(focus="all", tracker=tracker)

        table = Table(title=f"Architectural Insights (iter {iteration})")
        table.add_column("Severity", style="cyan", width=12)
        table.add_column("Component", width=20)
        table.add_column("Insight")
        table.add_column("Suggestion")

        for insight in analysis.insights:
            color = {"critical": "red", "improvement": "yellow", "minor": "dim"}.get(insight.severity, "white")
            table.add_row(
                f"[{color}]{insight.severity}[/]",
                insight.component,
                insight.insight[:60],
                insight.suggested_change[:60],
            )
        console.print(table)
        console.print(f"[bold]Bottleneck:[/] {analysis.bottleneck}")
        console.print(f"[bold]Top improvement:[/] {analysis.top_improvement}")

        # ── Phase 2: Generate patches ──
        console.print(f"\n[bold]Phase 2: Generating code patches...[/]")
        patches = generate_patches(analysis, tracker)

        if not patches.patches:
            console.print(f"  [dim]No actionable patches generated. Stopping.[/]")
            break

        for p in patches.patches:
            risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(p.risk, "white")
            console.print(f"  [{risk_color}][{p.risk}][/] {p.file}: {p.reason}")

        if not apply:
            console.print(f"\n  [dim]DRY RUN — patches not applied. Use apply=True for live mode.[/]")
            history.append({
                "iteration": iteration,
                "insights": len(analysis.insights),
                "patches": len(patches.patches),
                "applied": False,
            })
            continue

        # ── Phase 3: Benchmark BEFORE ──
        console.print(f"\n[bold]Phase 3: Benchmarking BEFORE...[/]")
        before = run_benchmark(benchmark_data, depth=depth)
        if before.error:
            console.print(f"  [red]Benchmark error: {before.error}[/]")
            break
        console.print(
            f"  Before: {before.n_principles} principles, "
            f"{before.confidence:.0%} conf, ${before.cost:.2f}"
        )

        # ── Phase 4: Apply patches ──
        console.print(f"\n[bold]Phase 4: Applying patches...[/]")
        applied_log, backup_paths = apply_patches(patches, max_risk=max_risk)
        for msg in applied_log:
            console.print(f"  {msg}")

        actual_applied = sum(1 for msg in applied_log if msg.startswith("APPLIED"))
        if actual_applied == 0:
            console.print(f"  [dim]No patches passed filters. Stopping.[/]")
            break

        # ── Phase 5: Benchmark AFTER ──
        console.print(f"\n[bold]Phase 5: Benchmarking AFTER...[/]")
        after = run_benchmark(benchmark_data, depth=depth)
        if after.error:
            console.print(f"  [red]Benchmark error after patches: {after.error}[/]")
            console.print(f"  [yellow]Rolling back...[/]")
            rollback(backup_paths)
            break

        console.print(
            f"  After: {after.n_principles} principles, "
            f"{after.confidence:.0%} conf, ${after.cost:.2f}"
        )

        # ── Phase 6: Keep or rollback ──
        # Improvement criteria: more principles OR higher confidence, without cost explosion
        improved = (
            (after.n_principles > before.n_principles) or
            (after.confidence > before.confidence + 0.02)
        ) and after.cost <= before.cost * 1.5

        if improved:
            console.print(f"  [green]✓ Improved! Keeping changes.[/]")
        else:
            console.print(f"  [yellow]✗ No improvement. Rolling back.[/]")
            rollback(backup_paths)

        history.append({
            "iteration": iteration,
            "insights": len(analysis.insights),
            "patches_generated": len(patches.patches),
            "patches_applied": actual_applied,
            "before": before.model_dump(),
            "after": after.model_dump(),
            "kept": improved,
        })

    # ── Save history ──
    META_HOME.mkdir(parents=True, exist_ok=True)
    log_path = META_HOME / f"meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps({
        "iterations": history,
        "total_cost": tracker.total_cost,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, ensure_ascii=False, default=str))

    console.print(f"\n[bold]💰 Meta-loop cost:[/] ${tracker.total_cost:.2f}")
    console.print(f"[dim]Log: {log_path}[/]")

    return {"history": history, "cost": tracker.total_cost}
