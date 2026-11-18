"""Aggregate trial results into summary reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import median
from typing import Any


def load_trials(path: Path) -> list[dict[str, Any]]:
    trials = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    phat = successes / n
    denom = 1 + (z**2 / n)
    centre = phat + (z**2 / (2 * n))
    margin = z * sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return [round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)]


def mean_interval(
    values: list[float],
    *,
    z: float = 1.96,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    centre = mean(values) or 0.0
    if len(values) == 1:
        return [round(centre, 4), round(centre, 4)]
    variance = sum((value - centre) ** 2 for value in values) / (len(values) - 1)
    margin = z * sqrt(variance / len(values))
    lower = centre - margin
    upper = centre + margin
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    return [round(lower, 4), round(upper, 4)]


def classify_error(error: Any) -> str | None:
    if not error:
        return None
    text = str(error).lower()
    if "not installed" in text or "no module named" in text:
        return "missing_dependency"
    if "api_key" in text or "authentication" in text or "unauthorized" in text:
        return "authentication"
    if "badrequest" in text or "unsupported" in text or "invalid request" in text:
        return "provider_request"
    if "json" in text or "parse" in text:
        return "parse_error"
    if "validation" in text or "schema" in text:
        return "schema_violation"
    if "retry" in text or "exceeded" in text:
        return "retry_exhaustion"
    return "runtime_error"


def rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def quality_values(group: list[dict[str, Any]]) -> list[float]:
    """Prefer explicit task-aware quality scores, with field_f1 as legacy fallback."""
    values = []
    for trial in group:
        if "quality_score" in trial:
            values.append(float(trial.get("quality_score", 0) or 0))
        elif "field_f1" in trial:
            values.append(float(trial.get("field_f1", 0) or 0))
    return values


def quality_metric_name(group: list[dict[str, Any]]) -> str:
    names = sorted({str(t.get("quality_metric")) for t in group if t.get("quality_metric")})
    if len(names) == 1:
        return names[0]
    if names:
        return "mixed:" + ",".join(names)
    return "field_f1"


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_fw_task: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        key = f"{t['framework']}::{t['task']}"
        by_fw_task[key].append(t)

    summary: dict[str, Any] = {}
    for key, group in by_fw_task.items():
        n = len(group)
        schema_successes = sum(1 for t in group if t.get("schema_valid"))
        zero_retry_successes = sum(
            1 for t in group if t.get("schema_valid") and int(t.get("retries", 0) or 0) == 0
        )
        retries = [float(t.get("retries", 0) or 0) for t in group]
        latencies = [float(t.get("latency_ms", 0) or 0) for t in group]
        token_totals = [
            float((t.get("tokens_in", 0) or 0) + (t.get("tokens_out", 0) or 0))
            for t in group
        ]
        field_f1s = [float(t.get("field_f1", 0)) for t in group if "field_f1" in t]
        qualities = quality_values(group)
        quality = mean(qualities)
        errors = [classify_error(t.get("error")) for t in group]
        error_counts: dict[str, int] = defaultdict(int)
        for error_type in errors:
            if error_type:
                error_counts[error_type] += 1

        field_score_values: dict[str, list[float]] = defaultdict(list)
        for trial in group:
            for field_name, value in (trial.get("field_scores") or {}).items():
                field_score_values[field_name].append(float(value))
        field_score_means = {
            field_name: round(sum(values) / len(values), 4)
            for field_name, values in sorted(field_score_values.items())
            if values
        }

        summary[key] = {
            "n": n,
            "schema_pass_rate": round(schema_successes / n, 4) if n else 0,
            "schema_pass_ci95": wilson_interval(schema_successes, n),
            "zero_retry_success_rate": round(zero_retry_successes / n, 4) if n else 0,
            "zero_retry_success_ci95": wilson_interval(zero_retry_successes, n),
            "mean_retries": rounded(mean(retries), 2) or 0,
            "median_retries": rounded(median(retries), 2) if retries else 0,
            "retry_rate": round(sum(1 for value in retries if value > 0) / n, 4) if n else 0,
            "mean_latency_ms": rounded(mean(latencies), 1) or 0,
            "median_latency_ms": rounded(median(latencies), 1) if latencies else 0,
            "p95_latency_ms": rounded(percentile(latencies, 0.95), 1) or 0,
            "mean_total_tokens": rounded(mean(token_totals), 1) or 0,
            "median_total_tokens": rounded(median(token_totals), 1) if token_totals else 0,
            "p95_total_tokens": rounded(percentile(token_totals, 0.95), 1) or 0,
            "mean_field_f1": rounded(mean(field_f1s), 4),
            "mean_quality_score": rounded(quality, 4),
            "quality_metric": quality_metric_name(group),
            "quality_ci95": mean_interval(
                qualities,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "field_score_means": field_score_means,
            "error_count": sum(error_counts.values()),
            "error_rate": round(sum(error_counts.values()) / n, 4) if n else 0,
            "error_type_counts": dict(sorted(error_counts.items())),
        }
    return summary


def write_summary(trials_path: Path, output_path: Path) -> dict[str, Any]:
    trials = load_trials(trials_path)
    summary = aggregate_trials(trials)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark trial records")
    parser.add_argument("trials_path", type=Path, help="Path to trials.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path; defaults to summary.recomputed.json beside trials",
    )
    args = parser.parse_args()

    output_path = args.output or args.trials_path.with_name("summary.recomputed.json")
    summary = write_summary(args.trials_path, output_path)
    print(f"Wrote {output_path}")
    for key, stats in summary.items():
        print(
            f"{key}: n={stats['n']} "
            f"schema_pass={stats['schema_pass_rate']} "
            f"quality={stats.get('mean_quality_score')} "
            f"errors={stats['error_count']}"
        )


if __name__ == "__main__":
    main()
