"""Python SDK — clean programmatic interface to Sparks.

Usage:
    from sparks import Sparks

    result = Sparks("Find the core principles").run("./data/")
    for p in result.principles:
        print(f"[{p.confidence:.0%}] {p.statement}")

    # With options
    sparks = Sparks("Find patterns", depth="deep", seed=42)
    result = sparks.run("./data/")

    # Full loop
    result = sparks.loop("./new_data/", cycles=3)

    # Self-optimize
    sparks.optimize(result)

    # Get explainability trace
    trace = sparks.last_trace
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sparks.state import SynthesisOutput


class Sparks:
    """Main entry point for the Sparks cognitive framework.

    Args:
        goal: What to discover in the data.
        depth: "quick" (4 tools), "standard" (7), or "deep" (13).
        seed: Random seed for reproducibility.
        backend: LLM backend ("cli", "anthropic", "openai", "google", "openai-compat").
        ablate: Dict of ablation flags (e.g. {"ablate_dopamine": True}).
        nervous: Whether to use the neural circuit (True) or sequential execution (False).
    """

    def __init__(
        self,
        goal: str,
        depth: str = "standard",
        seed: Optional[int] = None,
        backend: Optional[str] = None,
        ablate: Optional[dict[str, bool]] = None,
        nervous: bool = True,
    ):
        self.goal = goal
        self.depth = depth
        self.seed = seed
        self.ablate = ablate
        self.nervous = nervous
        self.last_trace: Optional[dict] = None
        self._result: Optional[SynthesisOutput] = None

        if backend:
            import os
            os.environ["SPARKS_BACKEND"] = backend

        if seed is not None:
            from sparks.research import set_seed
            set_seed(seed)

    def run(self, data_path: str | Path) -> SynthesisOutput:
        """Run cognitive analysis on data. Returns SynthesisOutput."""
        data_path = str(Path(data_path).resolve())

        if self.nervous:
            from sparks.autonomic import run_autonomic
            result = run_autonomic(
                goal=self.goal,
                data_path=data_path,
                depth=self.depth,
                ablate=self.ablate,
            )
        else:
            from sparks.engine import run as engine_run
            result = engine_run(
                goal=self.goal,
                data_path=data_path,
                depth=self.depth,
                nervous_system=False,
            )

        self._result = result
        self.last_trace = result.thinking_process.get("cascade_trace")
        return result

    def loop(
        self,
        data_path: str | Path,
        cycles: int = 1,
        budget: float = 10.0,
        predict: str = "",
        outcomes: str = "",
    ) -> None:
        """Run full loop (validate → evolve → predict → feedback).

        Requires a previous run() result or an output file.
        """
        if not self._result or not self._result.principles:
            raise ValueError("No principles to loop on. Call run() first.")

        from sparks.loop import run_loop
        from sparks.output import format_output

        # Save current result to temp file for loop input
        md = format_output(self._result, self.goal)
        tmp = Path.home() / ".sparks" / "tmp_loop_input.md"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(md)

        data_dirs = [str(Path(data_path).resolve())] if data_path else []
        run_loop(
            principles_source=str(tmp),
            data_dirs=data_dirs,
            cycles=cycles,
            budget=budget,
            predict_input=predict,
            outcomes=outcomes,
        )

    def optimize(self, result: Optional[SynthesisOutput] = None, apply: bool = False):
        """Self-optimize from analysis output."""
        target = result or self._result
        if not target:
            raise ValueError("No result to optimize from. Call run() first.")

        from sparks.output import format_output
        from sparks.self_optimize import self_optimize

        md = format_output(target, self.goal)
        tmp = Path.home() / ".sparks" / "tmp_optimize_input.md"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(md)

        self_optimize(output_path=str(tmp), apply=apply)

    def export_latex(self, result: Optional[SynthesisOutput] = None) -> str:
        """Export result as LaTeX table."""
        target = result or self._result
        if not target:
            raise ValueError("No result to export. Call run() first.")
        from sparks.research import to_latex_table
        return to_latex_table(target)

    def meta_improve(
        self,
        benchmark_data: str | Path,
        iterations: int = 3,
        apply: bool = False,
        max_risk: str = "low",
    ) -> dict:
        """Self-improvement loop: analyze own code → patch → benchmark → keep/rollback.

        Args:
            benchmark_data: Path to data for benchmarking.
            iterations: Max improvement cycles.
            apply: Actually apply changes (False = analysis only).
            max_risk: "low", "medium", or "high".
        """
        from sparks.meta import meta_loop
        return meta_loop(
            benchmark_data=str(Path(benchmark_data).resolve()),
            max_iterations=iterations,
            depth="quick",
            apply=apply,
            max_risk=max_risk,
        )

    def reset_circuit(self):
        """Reset circuit weights to initial values."""
        from sparks.circuit import NeuralCircuit
        c = NeuralCircuit()
        c.reset()
        c.save()

    @property
    def principles(self) -> list:
        """Quick access to last result's principles."""
        if self._result:
            return self._result.principles
        return []

    @property
    def cost(self) -> float:
        """Total cost of last run."""
        if self._result:
            return self._result.total_cost
        return 0.0

    def __repr__(self):
        return f"Sparks(goal={self.goal!r}, depth={self.depth!r})"
