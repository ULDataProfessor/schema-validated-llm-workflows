"""Simulation-based power analysis for instance-clustered paired contrasts."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values preserving input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def simulate_paired_power(
    *,
    n_instances: int,
    delta: float,
    sd: float,
    n_replicates: int = 3,
    alpha: float = 0.05,
    n_sim: int = 2000,
    seed: int = 42,
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    """Monte Carlo power for paired mean differences (framework vs SDK).

    When equivalence_margin is set, estimates power for TOST-style two one-sided
    tests of |delta_hat| < margin under the alternative that true delta=0.
    Otherwise estimates power to detect a non-zero mean difference `delta`.
    """
    rng = random.Random(seed)
    reject = 0
    for _ in range(n_sim):
        # Instance means after averaging nested replicates.
        diffs = []
        for _i in range(n_instances):
            replicate_diffs = [
                rng.gauss(delta, sd) for _r in range(max(1, n_replicates))
            ]
            diffs.append(sum(replicate_diffs) / len(replicate_diffs))
        mean_diff = sum(diffs) / len(diffs)
        var = sum((d - mean_diff) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
        se = math.sqrt(var / len(diffs)) if var > 0 else 1e-9
        t_stat = mean_diff / se
        # Normal approximation two-sided p-value.
        p = 2.0 * (1.0 - _phi(abs(t_stat)))
        if equivalence_margin is None:
            if p < alpha:
                reject += 1
        else:
            # TOST: both one-sided tests against +/- margin must reject.
            t_lower = (mean_diff + equivalence_margin) / se
            t_upper = (equivalence_margin - mean_diff) / se
            p_lower = 1.0 - _phi(t_lower)
            p_upper = 1.0 - _phi(t_upper)
            if max(p_lower, p_upper) < alpha:
                reject += 1
    return {
        "n_instances": n_instances,
        "n_replicates": n_replicates,
        "delta": delta,
        "sd": sd,
        "alpha": alpha,
        "n_sim": n_sim,
        "equivalence_margin": equivalence_margin,
        "power": reject / n_sim,
    }


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mde_for_power(
    *,
    n_instances: int,
    sd: float,
    target_power: float = 0.8,
    n_replicates: int = 3,
    alpha: float = 0.05,
    n_sim: int = 1500,
    seed: int = 42,
) -> float:
    """Binary-search approximate minimum detectable effect (paired mean)."""
    lo, hi = 0.0, max(sd * 3.0, 0.05)
    for _ in range(20):
        mid = (lo + hi) / 2.0
        est = simulate_paired_power(
            n_instances=n_instances,
            delta=mid,
            sd=sd,
            n_replicates=n_replicates,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
        )["power"]
        if est >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def build_analysis_plan(
    *,
    instance_counts: dict[str, int],
    n_replicates: int = 3,
    equivalence_margins: dict[str, float] | None = None,
    sd_assumptions: dict[str, float] | None = None,
) -> dict[str, Any]:
    equivalence_margins = equivalence_margins or {
        "schema_valid": 0.05,
        "quality_score": 0.05,
    }
    sd_assumptions = sd_assumptions or {
        "schema_valid": 0.20,
        "quality_score": 0.15,
    }
    tasks = {}
    for task, n_instances in instance_counts.items():
        task_entry: dict[str, Any] = {
            "n_instances": n_instances,
            "n_replicates": n_replicates,
            "primary_contrasts": ["each_condition_vs_plain_sdk"],
            "secondary_contrasts": ["pydantic_ai_vs_agno"],
            "multiplicity": "holm_within_task_vs_sdk",
            "metrics": {},
        }
        for metric, sd in sd_assumptions.items():
            margin = equivalence_margins.get(metric)
            power_diff = simulate_paired_power(
                n_instances=n_instances,
                delta=mde_for_power(
                    n_instances=n_instances,
                    sd=sd,
                    n_replicates=n_replicates,
                ),
                sd=sd,
                n_replicates=n_replicates,
            )
            power_equiv = simulate_paired_power(
                n_instances=n_instances,
                delta=0.0,
                sd=sd,
                n_replicates=n_replicates,
                equivalence_margin=margin,
            )
            task_entry["metrics"][metric] = {
                "sd_assumption": sd,
                "equivalence_margin": margin,
                "approx_mde_80": power_diff["delta"],
                "power_at_mde": power_diff["power"],
                "equivalence_power_at_zero": power_equiv["power"],
            }
        tasks[task] = task_entry
    return {
        "version": 1,
        "unit_of_analysis": "distinct_task_instance",
        "replicate_field": "seed",
        "cluster_key": ["task", "sample_id"],
        "alpha": 0.05,
        "bootstrap_resamples": 2000,
        "tasks": tasks,
        "estimands": {
            "structural": [
                "paired_delta_first_call_validity",
                "paired_delta_final_validity",
            ],
            "task": ["paired_delta_quality_score"],
            "cost": ["provider_calls", "total_tokens", "latency_ms"],
        },
    }


def default_instance_counts_from_trials_config(trials_cfg: dict[str, Any]) -> dict[str, int]:
    limits = trials_cfg.get("sample_limits", {})
    return {
        "extraction": int(limits.get("extraction", 100)),
        "sql_generation": int(limits.get("sql_generation", 50)),
        "summarization": int(limits.get("summarization", 50)),
        "compliance_qa": int(limits.get("compliance_qa", 50)),
        "multi_step": int(limits.get("multi_step", 50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/analysis_plan.json"),
        help="Path to write the frozen analysis plan JSON",
    )
    parser.add_argument("--trials-config", type=Path, default=Path("config/trials.yaml"))
    args = parser.parse_args()

    import yaml

    trials_cfg = yaml.safe_load(args.trials_config.read_text())
    counts = default_instance_counts_from_trials_config(trials_cfg)
    n_replicates = len(trials_cfg.get("seeds", [42, 43, 44]))
    plan = build_analysis_plan(instance_counts=counts, n_replicates=n_replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2))
    print(f"Wrote frozen analysis plan to {args.output}")


if __name__ == "__main__":
    main()
