"""Experiment Suite — systematic evaluation of Sparks.

Experiments:
1. Depth ablation: quick(4) vs standard(7) vs deep(13)
2. Circuit ablation: autonomic vs sequential (--no-nervous)
3. Reproducibility: 3 runs same config, measure consistency
4. Baseline: vanilla LLM (single prompt) vs Sparks pipeline

Run: python experiments/run_experiments.py --data ./demo/claude_code_posts/
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparks.research import set_seed, benchmark, format_benchmark, _mean, _std
from sparks.llm import llm_call
from sparks.cost import CostTracker, DEPTH_BUDGETS


def run_all(data_path: str, output_dir: str = "experiments/results"):
    """Run all experiments."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}

    # ════════════════════════════════════════
    # Experiment 1: Depth Ablation
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 1: Depth Ablation (quick vs standard vs deep)")
    print("="*60)

    for depth in ["quick", "standard"]:
        print(f"\n--- {depth} ---")
        set_seed(42)
        try:
            from sparks.autonomic import run_autonomic
            start = time.time()
            result = run_autonomic(goal="Extract the core principles from this data", data_path=data_path, depth=depth)
            elapsed = time.time() - start

            results[f"depth_{depth}"] = {
                "n_principles": len(result.principles),
                "confidence": result.confidence,
                "coverage": result.coverage,
                "cost": result.total_cost,
                "time": elapsed,
                "principles": [{"statement": p.statement, "confidence": p.confidence} for p in result.principles],
            }
            print(f"  Principles: {len(result.principles)}, Confidence: {result.confidence:.0%}, Cost: ${result.total_cost:.2f}, Time: {elapsed:.0f}s")
        except Exception as e:
            results[f"depth_{depth}"] = {"error": str(e)}
            print(f"  ERROR: {e}")

    # ════════════════════════════════════════
    # Experiment 2: Circuit Ablation
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 2: Circuit Ablation (autonomic vs sequential)")
    print("="*60)

    # Autonomic (already ran as standard above)
    if "depth_standard" in results and "error" not in results["depth_standard"]:
        results["circuit_on"] = results["depth_standard"]
        print(f"  Circuit ON: {results['circuit_on']['n_principles']} principles, {results['circuit_on']['confidence']:.0%}")

    # Sequential (--no-nervous)
    print("\n--- Sequential (no nervous system) ---")
    set_seed(42)
    try:
        from sparks.engine import run as engine_run
        start = time.time()
        result = engine_run(goal="Extract the core principles from this data", data_path=data_path, depth="standard", nervous_system=False)
        elapsed = time.time() - start

        results["circuit_off"] = {
            "n_principles": len(result.principles),
            "confidence": result.confidence,
            "coverage": result.coverage,
            "cost": result.total_cost,
            "time": elapsed,
            "principles": [{"statement": p.statement, "confidence": p.confidence} for p in result.principles],
        }
        print(f"  Circuit OFF: {len(result.principles)} principles, {result.confidence:.0%}, ${result.total_cost:.2f}, {elapsed:.0f}s")
    except Exception as e:
        results["circuit_off"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # ════════════════════════════════════════
    # Experiment 3: Baseline (vanilla LLM)
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 3: Baseline (single LLM call)")
    print("="*60)

    try:
        from sparks.data import DataStore
        data = DataStore(data_path)
        data_text = data.all_text(max_chars=80000)
        tracker = CostTracker(DEPTH_BUDGETS["standard"])

        start = time.time()
        baseline_result = llm_call(
            prompt=f"""Analyze this data and extract the core principles that govern it.
For each principle, provide:
- Statement
- Confidence (0-100%)
- Supporting evidence

Data:
{data_text}

Respond with clear, numbered principles.""",
            model="claude-opus-4-20250514" if os.environ.get("SPARKS_ALL_OPUS") else "claude-sonnet-4-20250514",
            tool="baseline",
            tracker=tracker,
        )
        elapsed = time.time() - start

        # Count principles (rough: count numbered lines)
        n_principles = sum(1 for line in baseline_result.split("\n") if line.strip() and line.strip()[0].isdigit() and "." in line[:5])

        results["baseline"] = {
            "n_principles": max(n_principles, 1),
            "cost": tracker.total_cost,
            "time": elapsed,
            "raw_output": baseline_result[:2000],
        }
        print(f"  Baseline: ~{n_principles} principles, ${tracker.total_cost:.2f}, {elapsed:.0f}s")
    except Exception as e:
        results["baseline"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # ════════════════════════════════════════
    # Experiment 4: Reproducibility (3 seeds)
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 4: Reproducibility (3 runs, different seeds)")
    print("="*60)

    repro_runs = []
    for seed in [42, 123, 777]:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        try:
            from sparks.autonomic import run_autonomic
            start = time.time()
            result = run_autonomic(goal="Extract the core principles from this data", data_path=data_path, depth="quick")
            elapsed = time.time() - start

            run_data = {
                "seed": seed,
                "n_principles": len(result.principles),
                "confidence": result.confidence,
                "cost": result.total_cost,
                "time": elapsed,
                "principles": [p.statement for p in result.principles],
            }
            repro_runs.append(run_data)
            print(f"  {len(result.principles)} principles, {result.confidence:.0%}, ${result.total_cost:.2f}")
        except Exception as e:
            repro_runs.append({"seed": seed, "error": str(e)})
            print(f"  ERROR: {e}")

    # Compute reproducibility
    successful = [r for r in repro_runs if "error" not in r]
    if len(successful) >= 2:
        from sparks.similarity import principle_convergence
        all_prins = [r["principles"] for r in successful]
        # Pairwise convergence
        convergences = []
        for i in range(len(all_prins)):
            for j in range(i+1, len(all_prins)):
                score, _ = principle_convergence(all_prins[i], all_prins[j])
                convergences.append(score)

        results["reproducibility"] = {
            "runs": repro_runs,
            "n_successful": len(successful),
            "mean_principles": _mean([r["n_principles"] for r in successful]),
            "std_principles": _std([r["n_principles"] for r in successful]),
            "mean_convergence": _mean(convergences) if convergences else 0,
            "total_cost": sum(r.get("cost", 0) for r in successful),
        }
        print(f"\n  Reproducibility: {_mean(convergences):.0%} cross-run convergence")
    else:
        results["reproducibility"] = {"runs": repro_runs, "error": "Not enough successful runs"}

    # ════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_cost = sum(
        r.get("cost", 0) for r in results.values() if isinstance(r, dict) and "cost" in r
    )
    if "reproducibility" in results and "runs" in results["reproducibility"]:
        total_cost += sum(r.get("cost", 0) for r in results["reproducibility"]["runs"] if isinstance(r, dict))

    print(f"\nTotal experiment cost: ${total_cost:.2f}")
    print(f"\n{'Experiment':<25} {'Principles':<12} {'Confidence':<12} {'Cost':<10}")
    print("-"*60)

    for key in ["depth_quick", "depth_standard", "circuit_on", "circuit_off", "baseline"]:
        r = results.get(key, {})
        if "error" in r:
            print(f"{key:<25} ERROR")
        elif "n_principles" in r:
            conf = f"{r.get('confidence', 0):.0%}" if "confidence" in r else "N/A"
            print(f"{key:<25} {r['n_principles']:<12} {conf:<12} ${r.get('cost', 0):.2f}")

    # Save
    save_path = out / f"experiment_{timestamp}.json"
    save_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="demo/claude_code_posts", help="Data path")
    parser.add_argument("--output", "-o", default="experiments/results", help="Output directory")
    args = parser.parse_args()

    run_all(data_path=args.data, output_dir=args.output)
