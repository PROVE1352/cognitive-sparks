"""CLI — sparks run --goal "..." --data ./path/ --depth standard"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="sparks",
    help="13 cognitive primitives that teach AI to think — not just compute.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    goal: str = typer.Option(..., "--goal", "-g", help="What to find in the data"),
    data: str = typer.Option(..., "--data", "-d", help="Path to data directory or file"),
    depth: str = typer.Option("standard", "--depth", help="quick / standard / deep"),
    output: str = typer.Option("", "--output", "-o", help="Output file path (default: stdout + ./output/)"),
    no_nervous: bool = typer.Option(False, "--no-nervous", help="Ablation: disable biological nervous system"),
):
    """Analyze data using 13 cognitive thinking tools."""
    from sparks.engine import run as engine_run
    from sparks.output import format_output

    if depth not in ("quick", "standard", "deep"):
        console.print(f"[red]Invalid depth: {depth}. Use quick/standard/deep.[/]")
        raise typer.Exit(1)

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Cognitive Analysis")
    console.print(f"[bold]Goal:[/] {goal}")
    console.print(f"[bold]Depth:[/] {depth}")

    if not no_nervous:
        # Autonomic mode: circuit-driven cascade (no TOOL_ORDER)
        from sparks.autonomic import run_autonomic
        result = run_autonomic(goal=goal, data_path=str(data_path), depth=depth)
    else:
        result = engine_run(goal=goal, data_path=str(data_path), depth=depth, nervous_system=False)
    md = format_output(result, goal)

    # Save to file
    out_path = output or "output/results.md"
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md)
    console.print(f"\n[bold green]📄 Results saved to {out_file}[/]")

    # Print key results
    console.print(f"\n{'='*60}")
    for i, p in enumerate(result.principles, 1):
        console.print(f"[bold]{i}. [{p.confidence:.0%}][/] {p.statement}")
    console.print(f"{'='*60}")


@app.command()
def evolve(
    goal: str = typer.Option(..., "--goal", "-g", help="What to find in the data"),
    data: str = typer.Option(..., "--data", "-d", help="Path to data directory or file"),
    generations: int = typer.Option(3, "--generations", "-n", help="Evolution generations"),
    output: str = typer.Option("", "--output", "-o", help="Output file path"),
):
    """Evolve principles through multiple generations (AutoAgent-style)."""
    from sparks.evolution import EvolutionLoop
    from sparks.output import format_output

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Evolution Mode")

    loop = EvolutionLoop(goal=goal, data_path=str(data_path), max_generations=generations)
    result = loop.run()
    loop.save_history()

    md = format_output(result, goal)
    out_path = output or "output/evolved_results.md"
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md)
    console.print(f"\n[bold green]📄 Evolved results saved to {out_file}[/]")

    console.print(f"\n{'='*60}")
    for i, p in enumerate(result.principles, 1):
        console.print(f"[bold]{i}. [{p.confidence:.0%}][/] {p.statement}")
    console.print(f"{'='*60}")


@app.command()
def loop(
    principles: str = typer.Option(..., "--principles", "-p", help="Path to .md output or store name"),
    data: str = typer.Option("", "--data", "-d", help="New data directory for validation"),
    cycles: int = typer.Option(1, "--cycles", "-n", help="B→C validation cycles"),
    budget: float = typer.Option(10.0, "--budget", "-b", help="Max cost in dollars"),
    predict: str = typer.Option("", "--predict", help="Situation to generate predictions for"),
    outcomes: str = typer.Option("", "--outcomes", help="Actual outcomes for feedback"),
):
    """Full loop: Validate → Evolve → Predict → Feedback."""
    from sparks.loop import run_loop

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Full Loop (B→F)")

    data_dirs = [data] if data else []
    run_loop(
        principles_source=principles,
        data_dirs=data_dirs,
        cycles=cycles,
        budget=budget,
        predict_input=predict,
        outcomes=outcomes,
    )


@app.command()
def info():
    """Show framework info."""
    from sparks import __version__
    console.print(f"[bold cyan]⚡ Sparks[/] v{__version__}")
    console.print("13 cognitive primitives for deep understanding.")
    console.print("\nBased on: Sparks of Genius (Root-Bernstein, 1999)")
    console.print("Architecture: 3-Layer Cognitive Harness")
    console.print("  Layer 0: Nervous System (sense, don't command)")
    console.print("  Layer 1: 13 Thinking Tools (observe → synthesize)")
    console.print("  Layer 2: AI Augmentation (forget, hold contradictions)")


if __name__ == "__main__":
    app()
