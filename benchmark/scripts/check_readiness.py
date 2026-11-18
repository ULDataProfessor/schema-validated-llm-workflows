#!/usr/bin/env python3
"""Check whether the benchmark is ready for a live matrix run."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

TASK_FILES = {
    "extraction": "extraction.jsonl",
    "sql_generation": "sql.jsonl",
    "summarization": "summarization.jsonl",
    "compliance_qa": "compliance_qa.jsonl",
    "multi_step": "multi_step.jsonl",
}

PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY", "API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

FRAMEWORK_IMPORTS = {
    "pydantic_ai": ("pydantic_ai",),
    "plain_sdk": ("openai",),
    "langgraph": ("langgraph", "langchain_openai"),
    "dspy": ("dspy",),
    "crewai": ("crewai",),
    "agno": ("agno",),
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def dotenv_keys() -> set[str]:
    keys: set[str] = set()
    for env_file in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if value.strip():
                keys.add(key.strip())
    return keys


def key_present(candidates: tuple[str, ...], file_keys: set[str]) -> bool:
    return any(os.environ.get(key) or key in file_keys for key in candidates)


def model_provider(model: str) -> str | None:
    if ":" not in model:
        return None
    return model.split(":", 1)[0]


def missing_framework_dependencies(frameworks: list[str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for framework in frameworks:
        imports = FRAMEWORK_IMPORTS.get(framework, ())
        missing_imports = [
            import_name
            for import_name in imports
            if importlib.util.find_spec(import_name) is None
        ]
        if missing_imports:
            missing[framework] = missing_imports
    return missing


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def selected_limits(trials: dict[str, Any], *, pilot: bool) -> dict[str, int]:
    if pilot:
        return {
            task: int(limit)
            for task, limit in trials.get("pilot", {}).get("sample_limits", {}).items()
        }
    return {task: int(limit) for task, limit in trials.get("sample_limits", {}).items()}


def build_report(*, pilot: bool, dry_run: bool, all_seeds: bool) -> tuple[list[str], list[str]]:
    models = load_yaml(ROOT / "config" / "models.yaml")
    trials = load_yaml(ROOT / "config" / "trials.yaml")
    limits = selected_limits(trials, pilot=pilot)
    tasks = list(trials["tasks"])
    frameworks = ["dry_run"] if dry_run else list(trials["frameworks"])
    seeds = list(trials["seeds"]) if (all_seeds or not pilot) else [trials["seeds"][0]]
    file_keys = dotenv_keys()
    errors: list[str] = []
    lines: list[str] = []

    mode = "dry-run" if dry_run else "live"
    scale = "pilot" if pilot else "full"
    lines.append(f"Mode: {mode} {scale}")
    lines.append(f"Frameworks: {len(frameworks)} ({', '.join(frameworks)})")
    lines.append(f"Tasks: {len(tasks)} ({', '.join(tasks)})")
    lines.append(f"Seeds: {len(seeds)} ({', '.join(str(seed) for seed in seeds)})")

    if not dry_run:
        provider = model_provider(str(models["default"]["model"]))
        if provider in PROVIDER_KEYS:
            present = key_present(PROVIDER_KEYS[provider], file_keys)
            state = "present" if present else "missing"
            lines.append(f"{provider} key: {state}")
            if not present:
                errors.append(f"Missing API key for provider '{provider}'")
        else:
            lines.append("provider key: not checked for unrecognized model provider")
        missing_deps = missing_framework_dependencies(frameworks)
        if missing_deps:
            lines.append("Framework dependencies: missing")
            for framework, imports in sorted(missing_deps.items()):
                joined = ", ".join(imports)
                lines.append(f"  {framework}: missing imports {joined}")
                errors.append(
                    f"{framework} is missing imports: {joined}. "
                    "Install optional framework extras in a supported Python environment."
                )
            version = ".".join(str(part) for part in sys.version_info[:3])
            lines.append(f"Python version: {version}")
        else:
            lines.append("Framework dependencies: ok")
    else:
        lines.append("provider key: not required for dry-run mode")

    total_per_framework_seed = 0
    lines.append("Data limits:")
    for task in tasks:
        limit = int(limits.get(task, 0))
        path = ROOT / "data" / "synthetic" / TASK_FILES[task]
        available = count_jsonl(path)
        total_per_framework_seed += limit
        state = "ok" if available >= limit else "short"
        lines.append(f"  {task}: limit={limit}, available={available}, {state}")
        if available < limit:
            errors.append(
                f"{task} has {available} records but the configured limit is {limit}"
            )

    planned_trials = len(frameworks) * len(seeds) * total_per_framework_seed
    lines.append(f"Planned trial rows: {planned_trials}")
    if not dry_run:
        lines.append("Note: live runs may issue one or more provider calls per trial because of retries.")

    return lines, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check benchmark live-run readiness")
    parser.add_argument("--pilot", action="store_true", help="Check pilot sample limits")
    parser.add_argument("--dry-run", action="store_true", help="Do not require provider keys")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Use all configured seeds for pilot mode; full mode always uses all seeds",
    )
    args = parser.parse_args()

    lines, errors = build_report(
        pilot=args.pilot,
        dry_run=args.dry_run,
        all_seeds=args.all_seeds,
    )
    print("Benchmark readiness report")
    print("==========================")
    for line in lines:
        print(line)
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nReady.")


if __name__ == "__main__":
    main()
