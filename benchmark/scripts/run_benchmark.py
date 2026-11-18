#!/usr/bin/env python3
"""Run benchmark trials across frameworks and tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import os

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    for env_file in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    if os.environ.get("API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.report import write_summary  # noqa: E402
from eval.metrics import (  # noqa: E402
    compliance_claim_grounding_scores,
    field_accuracy,
    required_fields_present,
    score_sql_result,
    summarization_quality_scores,
    workflow_execution_scores,
)
from eval.schema_boundary import revalidate_output  # noqa: E402
from frameworks import get_runner  # noqa: E402


TASK_FILES = {
    "extraction": "extraction.jsonl",
    "sql_generation": "sql.jsonl",
    "summarization": "summarization.jsonl",
    "compliance_qa": "compliance_qa.jsonl",
    "multi_step": "multi_step.jsonl",
}

RESUME_META_KEYS = (
    "model",
    "max_retries",
    "seeds",
    "dry_run",
    "pilot",
    "frameworks",
    "tasks",
    "task_limits",
)

PACKAGE_VERSION_NAMES = {
    "pydantic": "pydantic",
    "pydantic_ai": "pydantic-ai",
    "langgraph": "langgraph",
    "dspy": "dspy",
    "crewai": "crewai",
    "agno": "agno",
    "openai": "openai",
    "anthropic": "anthropic",
    "pyyaml": "PyYAML",
    "python_dotenv": "python-dotenv",
}

HASHED_CONFIG_FILES = (
    "pyproject.toml",
    "config/models.yaml",
    "config/trials.yaml",
    "config/ablations.yaml",
    "config/stress.yaml",
    "config/analysis_plan.json",
)

HASHED_SCORER_FILES = (
    "eval/metrics.py",
    "eval/schema_boundary.py",
    "scripts/run_benchmark.py",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def provenance_snapshot(args: argparse.Namespace) -> dict[str, object]:
    return {
        "started_at_utc": utc_timestamp(),
        "working_directory": str(Path.cwd()),
        "command_argv": list(sys.argv),
        "args": {
            key: jsonable(value)
            for key, value in sorted(vars(args).items())
        },
    }


def make_run_id(now: datetime | None = None) -> str:
    """Return a result directory id that is stable to sort and unique per process."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def trial_identity(*, framework: str, task: str, sample_id: str, seed: int) -> str:
    return f"{framework}::{task}::{sample_id}::{seed}"


def record_identity(record: dict) -> str:
    return trial_identity(
        framework=str(record["framework"]),
        task=str(record["task"]),
        sample_id=str(record["sample_id"]),
        seed=int(record["seed"]),
    )


def load_existing_trial_keys(trials_path: Path) -> set[str]:
    if not trials_path.exists():
        return set()
    keys = set()
    with trials_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            keys.add(record_identity(json.loads(line)))
    return keys


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def append_trial_record(trials_path: Path, record: dict) -> None:
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_meta(out_dir: Path, meta: dict) -> None:
    tmp_path = out_dir / "meta.json.tmp"
    tmp_path.write_text(json.dumps(meta, indent=2))
    tmp_path.replace(out_dir / "meta.json")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for key, package_name in PACKAGE_VERSION_NAMES.items():
        try:
            versions[key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def git_metadata() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def config_hashes() -> dict[str, str | None]:
    return {
        relative_path: sha256_file(ROOT / relative_path)
        for relative_path in HASHED_CONFIG_FILES
    }


def scorer_hashes() -> dict[str, str | None]:
    return {
        relative_path: sha256_file(ROOT / relative_path)
        for relative_path in HASHED_SCORER_FILES
    }


def environment_snapshot() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "package_versions": package_versions(),
        "git": git_metadata(),
        "config_hashes": config_hashes(),
        "scorer_hashes": scorer_hashes(),
    }


def counterbalanced_framework_order(
    frameworks: list[str],
    *,
    sample_id: str,
    seed: int,
) -> list[str]:
    """Deterministic per-instance permutation of condition order."""
    import hashlib

    digest = hashlib.sha256(f"{sample_id}:{seed}".encode()).hexdigest()
    offset = int(digest[:8], 16) % max(len(frameworks), 1)
    return frameworks[offset:] + frameworks[:offset]


def validate_resume_meta(out_dir: Path, meta: dict) -> None:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return
    existing = json.loads(meta_path.read_text())
    mismatches = []
    for key in RESUME_META_KEYS:
        if existing.get(key) != meta.get(key):
            mismatches.append((key, existing.get(key), meta.get(key)))
    if mismatches:
        rendered = "; ".join(
            f"{key}: existing={old!r}, requested={new!r}"
            for key, old, new in mismatches
        )
        raise ValueError(f"Cannot resume result directory with different configuration: {rendered}")


def load_config() -> tuple[dict, dict]:
    models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    trials = yaml.safe_load((ROOT / "config" / "trials.yaml").read_text())
    return models["default"], trials


def configured_task_limits(
    trial_cfg: dict,
    tasks: list[str],
    *,
    all_tasks: bool,
    pilot: bool,
    override_limit: int | None,
) -> dict[str, int | None]:
    if override_limit is not None:
        return {task: override_limit for task in tasks}

    configured_limits = (
        trial_cfg.get("pilot", {}).get("sample_limits", {})
        if pilot
        else trial_cfg.get("sample_limits", {})
    )
    if all_tasks or pilot:
        return {task: configured_limits.get(task) for task in tasks}
    return {task: None for task in tasks}


def load_task_samples(task: str, limit: int | None = None) -> list[dict]:
    if task not in TASK_FILES:
        raise ValueError(f"Unknown task: {task}")
    path = ROOT / "data" / "synthetic" / TASK_FILES[task]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/generate_synthetic_data.py"
        )
    samples = []
    with path.open() as f:
        for line in f:
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def _output_dict(output: object) -> dict:
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump(mode="json", exclude_none=True)
    if isinstance(output, dict):
        return output
    return {}


def score_task_result(task: str, result, sample: dict) -> dict[str, float]:
    gold = sample.get("gold", {})
    predicted = _output_dict(result.output)
    if result.output is None:
        if task == "sql_generation":
            return {
                "read_only": 0.0,
                "syntax_valid": 0.0,
                "execution_success": 0.0,
                "result_correct": 0.0,
            }
        if task == "compliance_qa":
            return {
                "answer": 0.0,
                "citation_support": 0.0,
                "unsupported_claims": 0.0,
                "claim_grounding": 0.0,
            }
        if task == "multi_step":
            return {
                "request_classification": 0.0,
                "tool_used": 0.0,
                "requires_human_review": 0.0,
                "tool_execution": 0.0,
                "state_path": 0.0,
            }
        return {k: 0.0 for k in gold if k != "unsupported_forbidden_terms"}

    if task == "sql_generation":
        return score_sql_result(
            str(predicted.get("sql", "")),
            sample["gold_sql"],
            sample["setup_sql"],
        )

    if task == "summarization":
        scores = required_fields_present(
            predicted,
            [
                "headline",
                "revenue_summary",
                "key_risks",
                "strategic_recommendations",
                "outlook",
            ],
        )
        scores.update(summarization_quality_scores(predicted, gold))
        return scores

    if task == "compliance_qa":
        return compliance_claim_grounding_scores(predicted, gold, sample["document"])

    if task == "multi_step":
        return workflow_execution_scores(
            predicted,
            gold,
            tool_result=getattr(result, "tool_result", None),
            state_path=getattr(result, "state_path", None),
        )

    # Drop generator-only metadata keys from exact-match field scoring.
    score_gold = {
        key: value
        for key, value in gold.items()
        if key
        not in {
            "unsupported_forbidden_terms",
            "expected_tool_result",
            "expected_state_path",
        }
    }
    return field_accuracy(predicted, score_gold)


def apply_boundary_revalidation(task: str, result, sample: dict) -> None:
    """Overwrite schema_valid with harness-boundary validation; keep runner flag."""
    runner_valid = bool(result.schema_valid)
    valid, parsed, error = revalidate_output(task, result.output, sample=sample)
    result.schema_valid = valid
    if parsed is not None:
        result.output = parsed
    if not valid and result.error is None:
        result.error = f"boundary_revalidation_failed: {error}"
    # Stash runner-local validity for the trial record via a dynamic attribute.
    result.runner_schema_valid = runner_valid  # type: ignore[attr-defined]


async def run_extraction_trials(
    framework: str,
    *,
    model: str,
    max_retries: int,
    seed: int,
    limit: int | None,
    dry_run: bool,
) -> list[dict]:
    runner = get_runner(framework, model=model, max_retries=max_retries, dry_run=dry_run)
    samples = load_task_samples("extraction", limit)
    records = []

    for sample in samples:
        subtype = sample.get("subtype", "invoice")
        result = await runner.run_extraction(sample["text"], subtype=subtype)

        if hasattr(runner, "score_extraction") and sample.get("gold"):
            result.field_scores = runner.score_extraction(result.output, sample["gold"])
        elif sample.get("gold") and result.output is not None:
            from eval.metrics import field_accuracy

            result.field_scores = field_accuracy(
                result.output.model_dump(mode="json"), sample["gold"]
            )
        elif sample.get("gold"):
            result.field_scores = {k: 0.0 for k in sample["gold"]}

        records.append(
            runner.to_trial_record(
                task="extraction",
                sample_id=sample["id"],
                seed=seed,
                result=result,
            )
        )
    return records


async def run_generic_task_trials(
    framework: str,
    task: str,
    *,
    model: str,
    max_retries: int,
    seed: int,
    limit: int | None,
    dry_run: bool,
) -> list[dict]:
    runner = get_runner(framework, model=model, max_retries=max_retries, dry_run=dry_run)
    samples = load_task_samples(task, limit)
    method_name = {
        "sql_generation": "run_sql_generation",
        "summarization": "run_summarization",
        "compliance_qa": "run_compliance_qa",
        "multi_step": "run_multi_step",
    }[task]
    records = []

    for sample in samples:
        try:
            result = await getattr(runner, method_name)(sample)
        except NotImplementedError as exc:
            print(f"SKIP: {exc}")
            return []
        result.field_scores = score_task_result(task, result, sample)
        records.append(
            runner.to_trial_record(
                task=task,
                sample_id=sample["id"],
                seed=seed,
                result=result,
            )
        )
    return records


async def run_task_sample(
    runner,
    *,
    task: str,
    sample: dict,
    seed: int,
    condition_order: int | None = None,
    ablation_id: str | None = None,
    ablation_condition: str | None = None,
    stress_id: str | None = None,
) -> dict | None:
    if task == "extraction":
        subtype = sample.get("subtype", "invoice")
        result = await runner.run_extraction(sample["text"], subtype=subtype)
        apply_boundary_revalidation(task, result, sample)

        if hasattr(runner, "score_extraction") and sample.get("gold"):
            result.field_scores = runner.score_extraction(result.output, sample["gold"])
        elif sample.get("gold") and result.output is not None:
            score_gold = {
                key: value
                for key, value in sample["gold"].items()
                if key not in {"unsupported_forbidden_terms"}
            }
            result.field_scores = field_accuracy(
                result.output.model_dump(mode="json"), score_gold
            )
        elif sample.get("gold"):
            result.field_scores = {
                k: 0.0
                for k in sample["gold"]
                if k not in {"unsupported_forbidden_terms"}
            }
    else:
        method_name = {
            "sql_generation": "run_sql_generation",
            "summarization": "run_summarization",
            "compliance_qa": "run_compliance_qa",
            "multi_step": "run_multi_step",
        }[task]
        try:
            result = await getattr(runner, method_name)(sample)
        except NotImplementedError as exc:
            print(f"SKIP: {exc}")
            return None
        apply_boundary_revalidation(task, result, sample)
        result.field_scores = score_task_result(task, result, sample)

    return runner.to_trial_record(
        task=task,
        sample_id=sample["id"],
        seed=seed,
        result=result,
        template_id=sample.get("template_id"),
        runner_schema_valid=getattr(result, "runner_schema_valid", None),
        condition_order=condition_order,
        ablation_id=ablation_id,
        ablation_condition=ablation_condition,
        stress_id=stress_id,
    )


def planned_trial_count(
    *,
    frameworks: list[str],
    tasks: list[str],
    seeds: list[int],
    samples_by_task: dict[str, list[dict]],
) -> int:
    return sum(
        len(samples_by_task[task])
        for _seed in seeds
        for _framework in frameworks
        for task in tasks
    )


async def async_main(args: argparse.Namespace) -> int:
    _load_dotenv()
    model_cfg, trial_cfg = load_config()
    model = args.model or model_cfg["model"]
    max_retries = args.max_retries or model_cfg["max_retries"]
    seeds = trial_cfg["seeds"] if args.all_seeds else [args.seed or trial_cfg["seeds"][0]]

    frameworks = trial_cfg["frameworks"] if args.all else [args.framework]
    tasks = trial_cfg["tasks"] if args.all else [args.task]
    task_limits = configured_task_limits(
        trial_cfg,
        tasks,
        all_tasks=args.all,
        pilot=args.pilot,
        override_limit=args.limit,
    )

    if args.dry_run:
        frameworks = ["dry_run"] if not args.all else ["dry_run"]

    samples_by_task = {
        task: load_task_samples(task, task_limits[task])
        for task in tasks
    }
    planned_trials = planned_trial_count(
        frameworks=frameworks,
        tasks=tasks,
        seeds=seeds,
        samples_by_task=samples_by_task,
    )

    run_id = args.run_id or make_run_id()
    results_root = Path(args.results_root)
    out_dir = results_root / run_id
    if args.resume:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    trials_path = out_dir / "trials.jsonl"

    meta = {
        "run_id": run_id,
        "model": model,
        "max_retries": max_retries,
        "environment": environment_snapshot(),
        "provenance": provenance_snapshot(args),
        "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": seeds,
        "dry_run": args.dry_run,
        "pilot": args.pilot,
        "frameworks": frameworks,
        "tasks": tasks,
        "task_limits": task_limits,
        "planned_trials": planned_trials,
        "n_trials": count_jsonl(trials_path) if args.resume else 0,
        "status": "running",
        "resumed": bool(args.resume),
    }
    if args.resume:
        validate_resume_meta(out_dir, meta)
    write_meta(out_dir, meta)

    existing_keys = load_existing_trial_keys(trials_path) if args.resume else set()
    skipped_existing = 0
    completed_this_run = 0
    runners = {
        fw: get_runner(
            fw,
            model=model,
            max_retries=max_retries,
            dry_run=args.dry_run,
        )
        for fw in frameworks
    }

    # Instance-major order with counterbalanced frameworks within each instance.
    for seed in seeds:
        for task in tasks:
            for sample in samples_by_task[task]:
                ordered = counterbalanced_framework_order(
                    frameworks,
                    sample_id=sample["id"],
                    seed=seed,
                )
                for order_idx, fw in enumerate(ordered):
                    identity = trial_identity(
                        framework=fw,
                        task=task,
                        sample_id=sample["id"],
                        seed=seed,
                    )
                    if identity in existing_keys:
                        skipped_existing += 1
                        continue
                    record = await run_task_sample(
                        runners[fw],
                        task=task,
                        sample=sample,
                        seed=seed,
                        condition_order=order_idx,
                    )
                    if record is None:
                        continue
                    append_trial_record(trials_path, record)
                    existing_keys.add(record_identity(record))
                    completed_this_run += 1
                    meta["n_trials"] = count_jsonl(trials_path)
                    write_meta(out_dir, meta)

    summary = write_summary(trials_path, out_dir / "summary.json")
    meta["n_trials"] = count_jsonl(trials_path)
    meta["status"] = "complete"
    meta["skipped_existing"] = skipped_existing
    meta["completed_this_run"] = completed_this_run
    meta["provenance"]["completed_at_utc"] = utc_timestamp()
    write_meta(out_dir, meta)

    print(f"\nBenchmark complete: {meta['n_trials']} trials")
    if args.resume:
        print(f"Resumed: skipped {skipped_existing} existing trials")
    print(f"Completed this run: {completed_this_run}")
    print(f"Results: {trials_path}")
    print(f"Summary: {out_dir / 'summary.json'}")
    for key, stats in summary.items():
        print(
            f"  {key}: schema_pass={stats['schema_pass_rate']}, "
            f"quality={stats.get('mean_quality_score')}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run type-safe agentic AI benchmark")
    parser.add_argument("--framework", default="pydantic_ai", help="Framework to run")
    parser.add_argument("--task", default="extraction", help="Task to run")
    parser.add_argument("--all", action="store_true", help="Run full framework x task matrix")
    parser.add_argument("--all-seeds", action="store_true", help="Run every configured seed")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use pilot sample limits from config/trials.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Mock runs without API calls")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per task")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Result directory name")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Directory that contains result run directories",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing result directory and skip completed trial keys",
    )
    args = parser.parse_args()
    if args.dry_run and args.limit is None and not args.pilot:
        args.limit = 5
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
