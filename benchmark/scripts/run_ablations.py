#!/usr/bin/env python3
"""Run labeled ablation / stress conditions separate from the primary matrix."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frameworks import get_runner  # noqa: E402
from scripts.run_benchmark import (  # noqa: E402
    _load_dotenv,
    append_trial_record,
    count_jsonl,
    environment_snapshot,
    load_config,
    load_task_samples,
    make_run_id,
    provenance_snapshot,
    run_task_sample,
    write_meta,
)
from eval.report import write_summary  # noqa: E402


def load_ablations() -> dict:
    return yaml.safe_load((ROOT / "config" / "ablations.yaml").read_text())


def load_stress() -> dict:
    return yaml.safe_load((ROOT / "config" / "stress.yaml").read_text())


def resolve_model(condition: str, primary: str, secondary: str) -> str:
    if condition == "secondary_model":
        return secondary
    return primary


def runner_kwargs_for_condition(condition: str) -> dict:
    kwargs: dict = {}
    if condition in {"validators_enabled", "typed_dependencies_and_tools"}:
        kwargs["enable_validators"] = condition == "validators_enabled"
        kwargs["use_typed_deps"] = condition == "typed_dependencies_and_tools"
    if condition == "zero_retry":
        kwargs["force_max_retries"] = 0
    return kwargs


async def run_ablation_matrix(args: argparse.Namespace) -> int:
    _load_dotenv()
    model_cfg, trial_cfg = load_config()
    ablations = load_ablations()["ablations"]
    primary_model = args.model or model_cfg["model"]
    secondary_model = model_cfg.get("secondary", {}).get(
        "model", "anthropic:claude-sonnet-4-20250514"
    )
    max_retries = args.max_retries or model_cfg["max_retries"]
    seeds = trial_cfg["seeds"][:1] if args.pilot else trial_cfg["seeds"]
    frameworks = trial_cfg["frameworks"] if args.all_frameworks else [args.framework]

    selected = [
        abl
        for abl in ablations
        if args.ablation_id is None or abl["id"] == args.ablation_id
    ]
    run_id = args.run_id or f"{make_run_id()}-ablations"
    out_dir = Path(args.results_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    trials_path = out_dir / "trials.jsonl"

    meta = {
        "run_id": run_id,
        "kind": "ablation",
        "model": primary_model,
        "secondary_model": secondary_model,
        "max_retries": max_retries,
        "environment": environment_snapshot(),
        "provenance": provenance_snapshot(args),
        "pilot": bool(args.pilot),
        "frameworks": frameworks,
        "ablation_ids": [a["id"] for a in selected],
        "status": "running",
        "n_trials": 0,
    }
    write_meta(out_dir, meta)

    completed = 0
    for abl in selected:
        for condition in abl["comparison"]:
            cond_name = condition["condition"]
            model = resolve_model(cond_name, primary_model, secondary_model)
            task_limits = abl.get("subset_limits") or {
                task: (10 if args.pilot else trial_cfg["sample_limits"].get(task))
                for task in abl["tasks"]
            }
            for fw in frameworks:
                rk = runner_kwargs_for_condition(cond_name)
                effective_retries = (
                    0 if rk.pop("force_max_retries", None) == 0 else max_retries
                )
                feature_kwargs = {
                    key: value
                    for key, value in rk.items()
                    if key in {"enable_validators", "use_typed_deps"}
                }
                # Only pydantic_ai currently accepts feature flags; others ignore via get_runner.
                try:
                    runner = get_runner(
                        fw,
                        model=model,
                        max_retries=effective_retries,
                        dry_run=args.dry_run,
                        **feature_kwargs,
                    )
                except TypeError:
                    runner = get_runner(
                        fw,
                        model=model,
                        max_retries=effective_retries,
                        dry_run=args.dry_run,
                    )
                    if hasattr(runner, "enable_validators") and "enable_validators" in feature_kwargs:
                        runner.enable_validators = feature_kwargs["enable_validators"]
                    if hasattr(runner, "use_typed_deps") and "use_typed_deps" in feature_kwargs:
                        runner.use_typed_deps = feature_kwargs["use_typed_deps"]

                for task in abl["tasks"]:
                    limit = task_limits.get(task)
                    if args.limit is not None:
                        limit = args.limit
                    elif args.pilot:
                        limit = min(limit or 10, 5)
                    samples = load_task_samples(task, limit)
                    for seed in seeds:
                        for sample in samples:
                            record = await run_task_sample(
                                runner,
                                task=task,
                                sample=sample,
                                seed=seed,
                                ablation_id=abl["id"],
                                ablation_condition=cond_name,
                            )
                            if record is None:
                                continue
                            append_trial_record(trials_path, record)
                            completed += 1
                            meta["n_trials"] = count_jsonl(trials_path)
                            write_meta(out_dir, meta)

    write_summary(trials_path, out_dir / "summary.json")
    meta["status"] = "complete"
    meta["n_trials"] = count_jsonl(trials_path)
    meta["provenance"]["completed_at_utc"] = meta["provenance"].get("started_at_utc")
    write_meta(out_dir, meta)
    print(f"Ablation run complete: {meta['n_trials']} trials in {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", default="plain_sdk")
    parser.add_argument("--all-frameworks", action="store_true")
    parser.add_argument("--ablation-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per task (overrides pilot/config limits).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run_ablation_matrix(args))


if __name__ == "__main__":
    raise SystemExit(main())
