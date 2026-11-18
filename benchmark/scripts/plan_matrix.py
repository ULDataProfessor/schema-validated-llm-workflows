#!/usr/bin/env python3
"""Report planned benchmark matrix size from the trial configuration."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskPlan:
    task: str
    sample_limit: int
    seeds: int
    frameworks: int
    n_per_framework_task: int
    planned_rows: int
    max_wilson_half_width: float


def load_trials(config_path: Path = ROOT / "config" / "trials.yaml") -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def selected_sample_limits(trials: dict[str, Any], *, pilot: bool) -> dict[str, int]:
    source = (
        trials.get("pilot", {}).get("sample_limits", {})
        if pilot
        else trials.get("sample_limits", {})
    )
    return {task: int(limit) for task, limit in source.items()}


def selected_seeds(trials: dict[str, Any], *, pilot: bool, all_seeds: bool) -> list[int]:
    seeds = [int(seed) for seed in trials.get("seeds", [])]
    if pilot and not all_seeds:
        return seeds[:1]
    return seeds


def selected_frameworks(trials: dict[str, Any], *, dry_run: bool) -> list[str]:
    return ["dry_run"] if dry_run else list(trials.get("frameworks", []))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + (z**2 / n)
    centre = phat + (z**2 / (2 * n))
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def max_wilson_half_width(n: int) -> float:
    """Approximate the widest 95% Wilson interval half-width for a binary rate."""
    if n <= 0:
        return 0.0
    successes = n // 2
    lower, upper = wilson_interval(successes, n)
    return (upper - lower) / 2


def build_plan(
    trials: dict[str, Any],
    *,
    pilot: bool = False,
    dry_run: bool = False,
    all_seeds: bool = False,
) -> list[TaskPlan]:
    limits = selected_sample_limits(trials, pilot=pilot)
    tasks = list(trials.get("tasks", []))
    frameworks = selected_frameworks(trials, dry_run=dry_run)
    seeds = selected_seeds(trials, pilot=pilot, all_seeds=all_seeds)

    plans = []
    for task in tasks:
        sample_limit = int(limits.get(task, 0))
        n_per_framework_task = sample_limit * len(seeds)
        planned_rows = n_per_framework_task * len(frameworks)
        plans.append(
            TaskPlan(
                task=task,
                sample_limit=sample_limit,
                seeds=len(seeds),
                frameworks=len(frameworks),
                n_per_framework_task=n_per_framework_task,
                planned_rows=planned_rows,
                max_wilson_half_width=max_wilson_half_width(n_per_framework_task),
            )
        )
    return plans


def format_plan(
    plans: list[TaskPlan],
    *,
    frameworks: list[str],
    seeds: list[int],
    pilot: bool,
    dry_run: bool,
) -> str:
    mode = "dry-run" if dry_run else "live"
    scale = "pilot" if pilot else "full"
    lines = [
        "Benchmark matrix plan",
        "=====================",
        f"Mode: {mode} {scale}",
        f"Frameworks: {len(frameworks)} ({', '.join(frameworks)})",
        f"Seeds: {len(seeds)} ({', '.join(str(seed) for seed in seeds)})",
        "",
        "Task                 Samples  n/group  Rows  Max 95% CI half-width",
        "-------------------  -------  -------  ----  ---------------------",
    ]
    for plan in plans:
        lines.append(
            f"{plan.task:<19}  "
            f"{plan.sample_limit:>7}  "
            f"{plan.n_per_framework_task:>7}  "
            f"{plan.planned_rows:>4}  "
            f"{plan.max_wilson_half_width * 100:>5.1f} pp"
        )
    lines.append("")
    lines.append(f"Total planned rows: {sum(plan.planned_rows for plan in plans)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report planned benchmark matrix size")
    parser.add_argument("--pilot", action="store_true", help="Use configured pilot limits")
    parser.add_argument("--dry-run", action="store_true", help="Use the dry-run framework only")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Use all configured seeds in pilot mode; full mode always uses all seeds",
    )
    args = parser.parse_args()

    trials = load_trials()
    all_seeds = args.all_seeds or not args.pilot
    frameworks = selected_frameworks(trials, dry_run=args.dry_run)
    seeds = selected_seeds(trials, pilot=args.pilot, all_seeds=all_seeds)
    plans = build_plan(
        trials,
        pilot=args.pilot,
        dry_run=args.dry_run,
        all_seeds=all_seeds,
    )
    print(
        format_plan(
            plans,
            frameworks=frameworks,
            seeds=seeds,
            pilot=args.pilot,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
