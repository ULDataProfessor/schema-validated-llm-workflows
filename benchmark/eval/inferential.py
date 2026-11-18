"""Paired framework comparisons from benchmark trial records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any

from eval.report import load_trials


PairKey = tuple[str, str, str]
PairRecord = tuple[PairKey, float, float]

LOWER_IS_BETTER = {"latency_ms", "retries", "total_tokens", "tokens_in", "tokens_out"}
BINARY_METRICS = {"schema_valid", "zero_retry_success"}


def trial_pair_key(trial: dict[str, Any]) -> PairKey:
    """Return the repeated-measures key used to pair frameworks."""
    try:
        task = trial["task"]
        sample_id = trial["sample_id"]
        seed = trial["seed"]
    except KeyError as exc:
        raise ValueError(f"Trial is missing required pairing field: {exc.args[0]}") from exc
    return str(task), str(sample_id), str(seed)


def metric_value(trial: dict[str, Any], metric: str) -> float:
    """Extract a numeric metric value from one trial record."""
    if metric in {"quality", "quality_score"}:
        if trial.get("quality_score") is not None:
            return float(trial["quality_score"])
        if trial.get("field_f1") is not None:
            return float(trial["field_f1"])
        raise ValueError("Trial is missing quality_score and field_f1")

    if metric == "field_f1":
        if trial.get("field_f1") is None:
            raise ValueError("Trial is missing field_f1")
        return float(trial["field_f1"])

    if metric == "schema_valid":
        return 1.0 if trial.get("schema_valid") else 0.0

    if metric == "zero_retry_success":
        retries = int(trial.get("retries", 0) or 0)
        return 1.0 if trial.get("schema_valid") and retries == 0 else 0.0

    if metric == "total_tokens":
        return float((trial.get("tokens_in", 0) or 0) + (trial.get("tokens_out", 0) or 0))

    if metric.startswith("field:"):
        field_name = metric.split(":", 1)[1]
        field_scores = trial.get("field_scores") or {}
        if field_name not in field_scores:
            raise ValueError(f"Trial is missing field score: {field_name}")
        return float(field_scores[field_name])

    if metric in {"latency_ms", "retries", "tokens_in", "tokens_out"}:
        if trial.get(metric) is None:
            raise ValueError(f"Trial is missing metric: {metric}")
        return float(trial[metric])

    raise ValueError(
        "Unsupported metric. Use quality_score, field_f1, schema_valid, "
        "zero_retry_success, latency_ms, retries, total_tokens, tokens_in, "
        "tokens_out, or field:<name>."
    )


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    return "higher_is_better"


def _index_trials(
    trials: list[dict[str, Any]],
    *,
    framework: str,
    task: str | None,
) -> dict[PairKey, dict[str, Any]]:
    indexed: dict[PairKey, dict[str, Any]] = {}
    for trial in trials:
        if trial.get("framework") != framework:
            continue
        if task is not None and trial.get("task") != task:
            continue
        key = trial_pair_key(trial)
        if key in indexed:
            raise ValueError(f"Duplicate trial for {framework} at {key}")
        indexed[key] = trial
    return indexed


def paired_records(
    trials: list[dict[str, Any]],
    *,
    framework_a: str,
    framework_b: str,
    metric: str = "quality_score",
    task: str | None = None,
) -> list[PairRecord]:
    """Return matched metric values for two frameworks on shared samples."""
    index_a = _index_trials(trials, framework=framework_a, task=task)
    index_b = _index_trials(trials, framework=framework_b, task=task)
    shared_keys = sorted(set(index_a) & set(index_b))
    return [
        (key, metric_value(index_a[key], metric), metric_value(index_b[key], metric))
        for key in shared_keys
    ]


def paired_differences(
    trials: list[dict[str, Any]],
    *,
    framework_a: str,
    framework_b: str,
    metric: str = "quality_score",
    task: str | None = None,
) -> list[float]:
    """Return A minus B differences for matched framework trials."""
    return [
        value_a - value_b
        for _, value_a, value_b in paired_records(
            trials,
            framework_a=framework_a,
            framework_b=framework_b,
            metric=metric,
            task=task,
        )
    ]


def binary_pair_counts(pairs: list[PairRecord]) -> dict[str, int]:
    """Return matched success/failure counts for binary paired metrics."""
    counts = {
        "a_success_b_failure": 0,
        "b_success_a_failure": 0,
        "both_success": 0,
        "both_failure": 0,
        "discordant_pairs": 0,
    }
    for _, value_a, value_b in pairs:
        if value_a not in {0.0, 1.0} or value_b not in {0.0, 1.0}:
            raise ValueError("Binary paired counts require metric values of 0.0 or 1.0")
        if value_a == 1.0 and value_b == 0.0:
            counts["a_success_b_failure"] += 1
            counts["discordant_pairs"] += 1
        elif value_a == 0.0 and value_b == 1.0:
            counts["b_success_a_failure"] += 1
            counts["discordant_pairs"] += 1
        elif value_a == 1.0 and value_b == 1.0:
            counts["both_success"] += 1
        else:
            counts["both_failure"] += 1
    return counts


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float],
    *,
    clusters: list[str] | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> list[float]:
    """Bootstrap a confidence interval for the mean paired difference."""
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        value = round(values[0], 4)
        return [value, value]
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = random.Random(seed)
    if clusters is None:
        sample_size = len(values)
        means = [
            mean(rng.choice(values) for _ in range(sample_size))
            for _ in range(n_resamples)
        ]
    else:
        if len(clusters) != len(values):
            raise ValueError("clusters must have the same length as values")
        cluster_values: dict[str, list[float]] = {}
        for cluster, value in zip(clusters, values, strict=True):
            cluster_values.setdefault(cluster, []).append(value)
        cluster_ids = sorted(cluster_values)
        if len(cluster_ids) == 1:
            value = round(mean(values), 4)
            return [value, value]

        means = []
        for _ in range(n_resamples):
            resampled_values: list[float] = []
            for _ in range(len(cluster_ids)):
                cluster_id = rng.choice(cluster_ids)
                resampled_values.extend(cluster_values[cluster_id])
            means.append(mean(resampled_values))

    lower = _percentile(means, alpha / 2)
    upper = _percentile(means, 1 - alpha / 2)
    return [round(lower, 4), round(upper, 4)]


def pair_clusters(pairs: list[PairRecord]) -> list[str]:
    """Cluster paired rows by task and sample, preserving seed-level repeats."""
    return [f"{key[0]}::{key[1]}" for key, _, _ in pairs]


def paired_summary(
    trials: list[dict[str, Any]],
    *,
    framework_a: str,
    framework_b: str,
    metric: str = "quality_score",
    task: str | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    cluster_bootstrap: bool = True,
) -> dict[str, Any]:
    pairs = paired_records(
        trials,
        framework_a=framework_a,
        framework_b=framework_b,
        metric=metric,
        task=task,
    )
    differences = [value_a - value_b for _, value_a, value_b in pairs]
    direction = metric_direction(metric)

    a_better = 0
    b_better = 0
    ties = 0
    for diff in differences:
        if diff == 0:
            ties += 1
        elif (direction == "higher_is_better" and diff > 0) or (
            direction == "lower_is_better" and diff < 0
        ):
            a_better += 1
        else:
            b_better += 1

    if not differences:
        mean_diff = None
        median_diff = None
        ci95 = [0.0, 0.0]
    else:
        mean_diff = round(mean(differences), 4)
        median_diff = round(median(differences), 4)
        ci95 = bootstrap_ci(
            differences,
            clusters=pair_clusters(pairs) if cluster_bootstrap else None,
            n_resamples=bootstrap_samples,
            seed=seed,
        )

    summary = {
        "framework_a": framework_a,
        "framework_b": framework_b,
        "task": task,
        "metric": metric,
        "direction": direction,
        "n_pairs": len(differences),
        "mean_a_minus_b": mean_diff,
        "median_a_minus_b": median_diff,
        "ci95_mean_a_minus_b": ci95,
        "a_better": a_better,
        "b_better": b_better,
        "ties": ties,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_cluster": "task_sample" if cluster_bootstrap else "trial_row",
        "n_clusters": len(set(pair_clusters(pairs))) if cluster_bootstrap else len(differences),
    }
    if metric in BINARY_METRICS:
        summary.update(binary_pair_counts(pairs))
    return summary


def write_paired_summary(
    trials_path: Path,
    output_path: Path,
    *,
    framework_a: str,
    framework_b: str,
    metric: str = "quality_score",
    task: str | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    cluster_bootstrap: bool = True,
) -> dict[str, Any]:
    trials = load_trials(trials_path)
    summary = paired_summary(
        trials,
        framework_a=framework_a,
        framework_b=framework_b,
        metric=metric,
        task=task,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        cluster_bootstrap=cluster_bootstrap,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired framework comparison")
    parser.add_argument("trials_path", type=Path, help="Path to trials.jsonl")
    parser.add_argument("--framework-a", required=True)
    parser.add_argument("--framework-b", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--metric", default="quality_score")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-cluster-bootstrap",
        action="store_true",
        help="Resample individual paired rows instead of task/sample clusters.",
    )
    args = parser.parse_args()

    trials = load_trials(args.trials_path)
    summary = paired_summary(
        trials,
        framework_a=args.framework_a,
        framework_b=args.framework_b,
        metric=args.metric,
        task=args.task,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        cluster_bootstrap=not args.no_cluster_bootstrap,
    )

    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
