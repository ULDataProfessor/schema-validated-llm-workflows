#!/usr/bin/env python3
"""Recompute task scores for existing trial records without provider calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.report import write_summary  # noqa: E402
from frameworks.base import FrameworkRunner  # noqa: E402
from scripts.run_benchmark import TASK_FILES, load_task_samples, score_task_result  # noqa: E402


def load_sample_index() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        task: {sample["id"]: sample for sample in load_task_samples(task)}
        for task in TASK_FILES
    }


def rescore_record(
    record: dict[str, Any],
    sample_index: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    task = record["task"]
    sample_id = record["sample_id"]
    sample = sample_index[task][sample_id]
    output = record.get("parsed_output")
    result = SimpleNamespace(output=output)
    field_scores = score_task_result(task, result, sample)
    field_f1 = (
        sum(field_scores.values()) / len(field_scores)
        if field_scores
        else (1.0 if record.get("schema_valid") else 0.0)
    )
    quality_score, quality_metric = FrameworkRunner._quality_score(
        task=task,
        field_scores=field_scores,
        schema_valid=bool(record.get("schema_valid")),
    )
    updated = dict(record)
    updated["field_scores"] = field_scores
    updated["field_f1"] = round(field_f1, 4)
    updated["quality_score"] = round(quality_score, 4)
    updated["quality_metric"] = quality_metric
    return updated


def rescore_trials(trials_path: Path, output_path: Path) -> None:
    sample_index = load_sample_index()
    with trials_path.open() as src, output_path.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            rescored = rescore_record(record, sample_index)
            dst.write(json.dumps(rescored) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute task scores for existing trials")
    parser.add_argument("trials_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    output_path = args.output or args.trials_path.with_name("trials.rescored.jsonl")
    if args.in_place:
        output_path = args.trials_path.with_suffix(".tmp")

    rescore_trials(args.trials_path, output_path)

    if args.in_place:
        output_path.replace(args.trials_path)
        output_path = args.trials_path

    summary_path = args.summary_output or output_path.with_name("summary.rescored.json")
    write_summary(output_path, summary_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
