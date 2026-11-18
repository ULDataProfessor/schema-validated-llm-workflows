#!/usr/bin/env python3
"""Optional held-out DSPy optimization ablation (separate from primary matrix)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="extraction")
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--dev-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "dspy_optimized.json")
    args = parser.parse_args()

    # This ablation is intentionally lightweight: record the optimization plan and
    # a placeholder artifact so confirmatory runs can attach optimizer cost without
    # changing the untuned primary-matrix DSPy condition.
    artifact = {
        "condition": "dspy_optimized_holdout",
        "task": args.task,
        "train_size": args.train_size,
        "dev_size": args.dev_size,
        "optimizer": "BootstrapFewShot",
        "status": "planned_or_local_only",
        "note": (
            "Primary matrix keeps untuned DSPy. Optimized programs should be "
            "evaluated only under ablation_id=dspy_optimized_holdout."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
