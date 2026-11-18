"""Estimate token-normalized quality and API cost from benchmark trials."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.report import load_trials, mean


def trial_quality(trial: dict[str, Any]) -> float:
    if trial.get("quality_score") is not None:
        return float(trial["quality_score"])
    if trial.get("field_f1") is not None:
        return float(trial["field_f1"])
    return 1.0 if trial.get("schema_valid") else 0.0


def trial_cost(
    trial: dict[str, Any],
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    tokens_in = float(trial.get("tokens_in", 0) or 0)
    tokens_out = float(trial.get("tokens_out", 0) or 0)
    return (
        tokens_in * input_cost_per_million / 1_000_000
        + tokens_out * output_cost_per_million / 1_000_000
    )


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def cost_per_success(total_cost: float | None, successes: int) -> float | None:
    if total_cost is None or successes <= 0:
        return None
    return total_cost / successes


def quality_per_1k_tokens(mean_quality: float | None, mean_tokens: float | None) -> float | None:
    if mean_quality is None or not mean_tokens:
        return None
    return mean_quality / (mean_tokens / 1000)


def quality_points_per_cost(quality_sum: float, total_cost: float | None) -> float | None:
    if total_cost is None or total_cost <= 0:
        return None
    return quality_sum / total_cost


def aggregate_costs(
    trials: list[dict[str, Any]],
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
    currency: str = "USD",
) -> dict[str, Any]:
    if input_cost_per_million < 0 or output_cost_per_million < 0:
        raise ValueError("Token prices must be non-negative")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        groups[f"{trial['framework']}::{trial['task']}"].append(trial)

    summary: dict[str, Any] = {
        "_pricing": {
            "currency": currency,
            "input_cost_per_million": input_cost_per_million,
            "output_cost_per_million": output_cost_per_million,
        }
    }
    for key, group in sorted(groups.items()):
        n = len(group)
        tokens_in = [float(trial.get("tokens_in", 0) or 0) for trial in group]
        tokens_out = [float(trial.get("tokens_out", 0) or 0) for trial in group]
        token_totals = [in_count + out_count for in_count, out_count in zip(tokens_in, tokens_out)]
        qualities = [trial_quality(trial) for trial in group]
        costs = [
            trial_cost(
                trial,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
            )
            for trial in group
        ]
        token_observed = sum(1 for value in token_totals if value > 0)
        total_cost = sum(costs) if token_observed else None
        quality_sum = sum(qualities)
        schema_successes = sum(1 for trial in group if trial.get("schema_valid"))
        zero_retry_successes = sum(
            1
            for trial in group
            if trial.get("schema_valid") and int(trial.get("retries", 0) or 0) == 0
        )
        mean_quality = mean(qualities)
        mean_tokens = mean(token_totals)

        summary[key] = {
            "n": n,
            "total_tokens_in": int(sum(tokens_in)),
            "total_tokens_out": int(sum(tokens_out)),
            "total_tokens": int(sum(token_totals)),
            "token_accounting_coverage_rate": round(token_observed / n, 4) if n else 0.0,
            "mean_total_tokens": rounded(mean_tokens, 2),
            "total_estimated_cost": rounded(total_cost, 8),
            "mean_estimated_cost": rounded(mean(costs), 8) if token_observed else None,
            "schema_valid_successes": schema_successes,
            "zero_retry_successes": zero_retry_successes,
            "cost_per_schema_valid_run": rounded(cost_per_success(total_cost, schema_successes), 8),
            "cost_per_zero_retry_success": rounded(
                cost_per_success(total_cost, zero_retry_successes),
                8,
            ),
            "mean_quality_score": rounded(mean_quality, 4),
            "quality_points": rounded(quality_sum, 4),
            "quality_per_1k_tokens": rounded(
                quality_per_1k_tokens(mean_quality, mean_tokens),
                4,
            ),
            "quality_points_per_cost": rounded(
                quality_points_per_cost(quality_sum, total_cost),
                4,
            ),
        }
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    pricing = summary["_pricing"]
    currency = pricing["currency"]
    lines = [
        "# Cost Analysis",
        "",
        f"- Currency: `{currency}`",
        f"- Input cost per million tokens: `{pricing['input_cost_per_million']}`",
        f"- Output cost per million tokens: `{pricing['output_cost_per_million']}`",
        "",
        "| Framework | Task | n | Token cov. | Tokens | Cost | Cost/valid | Quality | Q/1k tok | Q/cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, stats in summary.items():
        if key == "_pricing":
            continue
        framework, task = key.split("::", 1)
        lines.append(
            "| "
            + " | ".join(
                [
                    framework,
                    task,
                    str(stats["n"]),
                    f"{stats['token_accounting_coverage_rate'] * 100:.1f}%",
                    str(stats["total_tokens"]),
                    "--"
                    if stats["total_estimated_cost"] is None
                    else f"{stats['total_estimated_cost']:.8f}",
                    "--"
                    if stats["cost_per_schema_valid_run"] is None
                    else f"{stats['cost_per_schema_valid_run']:.8f}",
                    "--" if stats["mean_quality_score"] is None else f"{stats['mean_quality_score']:.4f}",
                    "--" if stats["quality_per_1k_tokens"] is None else f"{stats['quality_per_1k_tokens']:.4f}",
                    "--"
                    if stats["quality_points_per_cost"] is None
                    else f"{stats['quality_points_per_cost']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Costs are estimates derived from archived token counts and supplied pricing.",
            "They should be regenerated if provider prices, cached-token rules, or billing terms change.",
            "",
        ]
    )
    return "\n".join(lines)


def write_cost_analysis(
    trials_path: Path,
    *,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    input_cost_per_million: float,
    output_cost_per_million: float,
    currency: str = "USD",
) -> dict[str, Any]:
    trials = load_trials(trials_path)
    summary = aggregate_costs(
        trials,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        currency=currency,
    )
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(summary, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate benchmark token costs")
    parser.add_argument("trials_path", type=Path)
    parser.add_argument("--input-cost-per-million", type=float, required=True)
    parser.add_argument("--output-cost-per-million", type=float, required=True)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    summary = write_cost_analysis(
        args.trials_path,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
        currency=args.currency,
    )
    if not args.json_output and not args.markdown_output:
        print(render_markdown(summary))
    else:
        if args.json_output:
            print(f"Wrote {args.json_output}")
        if args.markdown_output:
            print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
