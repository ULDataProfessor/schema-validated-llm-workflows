#!/usr/bin/env python3
"""Run the configured pilot matrix before launching the full benchmark."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _env_files() -> list[Path]:
    return [ROOT / ".env", ROOT.parent / ".env"]


def load_dotenv() -> None:
    for env_file in _env_files():
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


def has_api_key() -> bool:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    return bool(key and len(key.strip()) > 3)


async def run_pilot(
    limit: int | None,
    dry_run: bool,
    all_seeds: bool,
    run_id: str | None,
    results_root: Path | None,
    resume: bool,
) -> int:
    load_dotenv()
    if not dry_run and not has_api_key():
        print("ERROR: No OPENAI_API_KEY found.")
        print("  Copy benchmark/.env.example to benchmark/.env and add your key,")
        print("  or run with --dry-run for pipeline validation.")
        return 1

    readiness_script = ROOT / "scripts" / "check_readiness.py"
    run_script = ROOT / "scripts" / "run_benchmark.py"

    readiness_cmd = [sys.executable, str(readiness_script), "--pilot"]
    if dry_run:
        readiness_cmd.append("--dry-run")
    if all_seeds:
        readiness_cmd.append("--all-seeds")
    readiness = subprocess.run(readiness_cmd, cwd=ROOT)
    if readiness.returncode != 0:
        return readiness.returncode

    cmd = [sys.executable, str(run_script), "--all", "--pilot"]
    if dry_run:
        cmd.append("--dry-run")
    if all_seeds:
        cmd.append("--all-seeds")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if run_id is not None:
        cmd.extend(["--run-id", run_id])
    if results_root is not None:
        cmd.extend(["--results-root", str(results_root)])
    if resume:
        cmd.append("--resume")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        print("\nPilot matrix complete.")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured pilot matrix")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run_pilot(
                args.limit,
                args.dry_run,
                args.all_seeds,
                args.run_id,
                args.results_root,
                args.resume,
            )
        )
    )


if __name__ == "__main__":
    main()
