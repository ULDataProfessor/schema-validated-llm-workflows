"""Project full-matrix token cost from an observed pilot run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from eval.report import load_trials, mean

ROOT = Path(__file__).resolve().parents[1]


def load_trial_config(path: Path = ROOT / "config" / "trials.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def target_sample_limits(config: dict[str, Any], *, target: str) -> dict[str, int]:
    if target == "pilot":
        source = config.get("pilot", {}).get("sample_limits", {})
    else:
        source = config.get("sample_limits", {})
    return {task: int(limit) for task, limit in source.items()}


def target_seed_count(config: dict[str, Any], *, target: str) -> int:
    seeds = list(config.get("seeds", []))
    if target == "pilot":
        return 1 if seeds else 0
    return len(seeds)


def trial_tokens(trial: dict[str, Any]) -> tuple[int, int]:
    return int(trial.get("tokens_in", 0) or 0), int(trial.get("tokens_out", 0) or 0)


def projected_cost(
    *,
    tokens_in: float,
    tokens_out: float,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    return (
        tokens_in * input_cost_per_million / 1_000_000
        + tokens_out * output_cost_per_million / 1_000_000
    )


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def build_projection(
    trials: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    input_cost_per_million: float,
    output_cost_per_million: float,
    currency: str = "USD",
    target: str = "full",
) -> dict[str, Any]:
    if input_cost_per_million < 0 or output_cost_per_million < 0:
        raise ValueError("Token prices must be non-negative")
    if target not in {"full", "pilot"}:
        raise ValueError("target must be 'full' or 'pilot'")

    frameworks = list(config.get("frameworks", []))
    tasks = list(config.get("tasks", []))
    limits = target_sample_limits(config, target=target)
    seed_count = target_seed_count(config, target=target)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        groups[(str(trial["framework"]), str(trial["task"]))].append(trial)

    cells: dict[str, dict[str, Any]] = {}
    total_projected_tokens_in = 0.0
    total_projected_tokens_out = 0.0
    total_projected_cost = 0.0
    unknown_cells: list[str] = []

    for framework in frameworks:
        for task in tasks:
            key = f"{framework}::{task}"
            group = groups.get((framework, task), [])
            observed_with_tokens = []
            for trial in group:
                tokens_in, tokens_out = trial_tokens(trial)
                if tokens_in + tokens_out > 0:
                    observed_with_tokens.append((tokens_in, tokens_out))

            target_rows = int(limits.get(task, 0)) * seed_count
            token_coverage_rate = (
                len(observed_with_tokens) / len(group) if group else 0.0
            )
            mean_tokens_in = mean([item[0] for item in observed_with_tokens])
            mean_tokens_out = mean([item[1] for item in observed_with_tokens])

            if mean_tokens_in is None or mean_tokens_out is None:
                unknown_cells.append(key)
                cells[key] = {
                    "framework": framework,
                    "task": task,
                    "observed_trials": len(group),
                    "observed_token_trials": len(observed_with_tokens),
                    "token_accounting_coverage_rate": round(token_coverage_rate, 4),
                    "target_rows": target_rows,
                    "mean_tokens_in": None,
                    "mean_tokens_out": None,
                    "projected_tokens_in": None,
                    "projected_tokens_out": None,
                    "projected_total_tokens": None,
                    "projected_cost": None,
                    "projection_status": "missing_token_accounting",
                }
                continue

            projected_tokens_in = mean_tokens_in * target_rows
            projected_tokens_out = mean_tokens_out * target_rows
            cell_cost = projected_cost(
                tokens_in=projected_tokens_in,
                tokens_out=projected_tokens_out,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
            )
            total_projected_tokens_in += projected_tokens_in
            total_projected_tokens_out += projected_tokens_out
            total_projected_cost += cell_cost

            cells[key] = {
                "framework": framework,
                "task": task,
                "observed_trials": len(group),
                "observed_token_trials": len(observed_with_tokens),
                "token_accounting_coverage_rate": round(token_coverage_rate, 4),
                "target_rows": target_rows,
                "mean_tokens_in": rounded(mean_tokens_in, 2),
                "mean_tokens_out": rounded(mean_tokens_out, 2),
                "projected_tokens_in": int(round(projected_tokens_in)),
                "projected_tokens_out": int(round(projected_tokens_out)),
                "projected_total_tokens": int(
                    round(projected_tokens_in + projected_tokens_out)
                ),
                "projected_cost": rounded(cell_cost, 8),
                "projection_status": "projected",
            }

    return {
        "_pricing": {
            "currency": currency,
            "input_cost_per_million": input_cost_per_million,
            "output_cost_per_million": output_cost_per_million,
        },
        "_target": {
            "target": target,
            "frameworks": frameworks,
            "tasks": tasks,
            "seed_count": seed_count,
            "sample_limits": limits,
            "planned_rows": sum(
                int(limits.get(task, 0)) * seed_count * len(frameworks)
                for task in tasks
            ),
        },
        "_totals": {
            "projected_cells": len(cells) - len(unknown_cells),
            "unknown_cells": unknown_cells,
            "known_projected_tokens_in": int(round(total_projected_tokens_in)),
            "known_projected_tokens_out": int(round(total_projected_tokens_out)),
            "known_projected_total_tokens": int(
                round(total_projected_tokens_in + total_projected_tokens_out)
            ),
            "known_projected_cost": rounded(total_projected_cost, 8),
        },
        "cells": cells,
    }


def render_markdown(projection: dict[str, Any]) -> str:
    pricing = projection["_pricing"]
    target = projection["_target"]
    totals = projection["_totals"]
    lines = [
        "# Matrix Cost Projection",
        "",
        f"- Target: `{target['target']}`",
        f"- Planned rows: `{target['planned_rows']}`",
        f"- Currency: `{pricing['currency']}`",
        f"- Input cost per million tokens: `{pricing['input_cost_per_million']}`",
        f"- Output cost per million tokens: `{pricing['output_cost_per_million']}`",
        f"- Known projected total tokens: `{totals['known_projected_total_tokens']}`",
        f"- Known projected cost: `{totals['known_projected_cost']}`",
        f"- Unknown cells: `{len(totals['unknown_cells'])}`",
        "",
        "| Framework | Task | Obs. | Token cov. | Target rows | Mean tok. | Projected tok. | Projected cost | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, cell in sorted(projection["cells"].items()):
        mean_tokens = (
            "--"
            if cell["mean_tokens_in"] is None
            else f"{cell['mean_tokens_in'] + cell['mean_tokens_out']:.2f}"
        )
        projected_tokens = (
            "--" if cell["projected_total_tokens"] is None else str(cell["projected_total_tokens"])
        )
        projected = (
            "--" if cell["projected_cost"] is None else f"{cell['projected_cost']:.8f}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    cell["framework"],
                    cell["task"],
                    str(cell["observed_trials"]),
                    f"{cell['token_accounting_coverage_rate'] * 100:.1f}%",
                    str(cell["target_rows"]),
                    mean_tokens,
                    projected_tokens,
                    projected,
                    cell["projection_status"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Projection uses observed token means from the supplied trial rows and scales them to the configured target matrix.",
            "Cells with missing token accounting are reported as unknown rather than estimated from other frameworks.",
            "These values are planning estimates, not substitutes for archived token counts from the completed run.",
            "",
        ]
    )
    return "\n".join(lines)


def write_projection(
    trials_path: Path,
    *,
    config_path: Path = ROOT / "config" / "trials.yaml",
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    input_cost_per_million: float,
    output_cost_per_million: float,
    currency: str = "USD",
    target: str = "full",
) -> dict[str, Any]:
    projection = build_projection(
        load_trials(trials_path),
        config=load_trial_config(config_path),
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        currency=currency,
        target=target,
    )
    projection["_source"] = {
        "trials_path": str(trials_path),
        "config_path": str(config_path),
    }
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(projection, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(projection))
    return projection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project target-matrix token cost from observed pilot trials"
    )
    parser.add_argument("trials_path", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--target", choices=["full", "pilot"], default="full")
    parser.add_argument("--input-cost-per-million", type=float, required=True)
    parser.add_argument("--output-cost-per-million", type=float, required=True)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    projection = write_projection(
        args.trials_path,
        config_path=args.config,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
        currency=args.currency,
        target=args.target,
    )
    if not args.json_output and not args.markdown_output:
        print(render_markdown(projection))
    else:
        if args.json_output:
            print(f"Wrote {args.json_output}")
        if args.markdown_output:
            print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
