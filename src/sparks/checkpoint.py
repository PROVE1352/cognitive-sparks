"""Checkpoint — save/restore state after each tool firing.

If a run crashes at tool #8, you don't lose tools #1-7.
Resume from last checkpoint instead of starting over.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from sparks.state import CognitiveState

CHECKPOINT_DIR = Path.home() / ".sparks" / "checkpoints"


def _write_checkpoint(path: Path, serialized: str) -> None:
    """Write checkpoint atomically (temp file + os.replace) to avoid partial reads on crash."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(serialized)
    os.replace(tmp, path)


class Checkpoint:
    """Saves CognitiveState to disk after each tool firing."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = CHECKPOINT_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self._last_write_thread: Optional[threading.Thread] = None

    def save(self, state: CognitiveState, tool_name: str, cost_so_far: float):
        """Save state after a tool firing."""
        self.step += 1
        data = {
            "step": self.step,
            "tool": tool_name,
            "cost": cost_so_far,
            "timestamp": datetime.now().isoformat(),
            "state": {
                "goal": state.goal,
                "depth": state.depth,
                "round": state.round,
                "phase": state.phase.value,
                "observations": [o.model_dump() for o in state.observations],
                "patterns": [p.model_dump() for p in state.patterns],
                "principles": [p.model_dump() for p in state.principles],
                "analogies": [a.model_dump() for a in state.analogies],
                "contradictions": [c.model_dump() for c in state.contradictions],
                "model_results": [m.model_dump() for m in state.model_results],
                "hypotheses": [h.model_dump() for h in getattr(state, 'hypotheses', [])],
                "perspective_insights": [p.model_dump() for p in getattr(state, 'perspective_insights', [])],
                "play_discoveries": [d.model_dump() for d in getattr(state, 'play_discoveries', [])],
            },
        }
        path = self.dir / f"step_{self.step:03d}_{tool_name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    @staticmethod
    def latest_run() -> Optional[str]:
        """Find the most recent run ID."""
        if not CHECKPOINT_DIR.exists():
            return None
        runs = sorted(CHECKPOINT_DIR.iterdir(), reverse=True)
        return runs[0].name if runs else None

    @staticmethod
    def restore(run_id: str) -> Optional[tuple[CognitiveState, int, float]]:
        """Restore state from the latest checkpoint of a run.

        Returns (state, step_number, cost_so_far) or None.
        """
        from sparks.state import (
            CognitiveState, Observation, Pattern, Principle, Analogy,
            Contradiction, ModelResult, Hypothesis, PerspectiveInsight,
            PlayDiscovery, Phase,
        )

        run_dir = CHECKPOINT_DIR / run_id
        if not run_dir.exists():
            return None

        # Find latest step
        steps = sorted(run_dir.glob("step_*.json"), reverse=True)
        if not steps:
            return None

        data = json.loads(steps[0].read_text())
        s = data["state"]

        state = CognitiveState(
            goal=s["goal"],
            depth=s.get("depth", "standard"),
            round=s.get("round", 0),
            phase=Phase(s.get("phase", "sequential")),
        )
        state.observations = [Observation.model_validate(o) for o in s.get("observations", [])]
        state.patterns = [Pattern.model_validate(p) for p in s.get("patterns", [])]
        state.principles = [Principle.model_validate(p) for p in s.get("principles", [])]
        state.analogies = [Analogy.model_validate(a) for a in s.get("analogies", [])]
        state.contradictions = [Contradiction.model_validate(c) for c in s.get("contradictions", [])]
        state.model_results = [ModelResult.model_validate(m) for m in s.get("model_results", [])]
        state.hypotheses = [Hypothesis.model_validate(h) for h in s.get("hypotheses", [])]
        state.perspective_insights = [PerspectiveInsight.model_validate(p) for p in s.get("perspective_insights", [])]
        state.play_discoveries = [PlayDiscovery.model_validate(d) for d in s.get("play_discoveries", [])]

        return state, data["step"], data["cost"]

    def flush(self):
        """Block until the last background checkpoint write completes. Call at cascade end."""
        if self._last_write_thread and self._last_write_thread.is_alive():
            self._last_write_thread.join()

    def cleanup(self):
        """Remove checkpoint files after successful completion."""
        import shutil
        self.flush()
        if self.dir.exists():
            shutil.rmtree(self.dir)
