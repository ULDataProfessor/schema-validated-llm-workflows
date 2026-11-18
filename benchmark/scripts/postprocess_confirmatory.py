#!/usr/bin/env python3
"""Post-process an eligible confirmatory result directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-framework", default="plain_sdk")
    args = parser.parse_args()
    run_dir = args.run_dir
    trials = run_dir / "trials.jsonl"
    if not trials.exists():
        raise SystemExit(f"missing {trials}")

    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("pilot"):
        print("WARNING: run is still flagged pilot=true")
    if meta.get("status") != "complete":
        print(f"WARNING: status={meta.get('status')!r}")

    # Human audit packet (blinded).
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "eval.human_audit",
            str(trials),
            "--json-output",
            str(run_dir / "human_audit_packet.json"),
            "--csv-output",
            str(run_dir / "human_audit_packet.csv"),
            "--markdown-output",
            str(run_dir / "human_audit_packet.md"),
            "--cases-per-task",
            "5",
            "--controls-per-task",
            "1",
        ],
        cwd=ROOT,
    )

    # Inferential contrasts vs SDK for each non-reference framework.
    frameworks = sorted(
        {
            json.loads(line)["framework"]
            for line in trials.read_text().splitlines()
            if line.strip()
        }
    )
    inferential_dir = run_dir / "inferential"
    inferential_dir.mkdir(exist_ok=True)
    for framework in frameworks:
        if framework == args.reference_framework:
            continue
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "eval.inferential",
                str(trials),
                "--framework-a",
                framework,
                "--framework-b",
                args.reference_framework,
                "--metric",
                "quality_score",
                "--output",
                str(inferential_dir / f"{framework}_vs_{args.reference_framework}.json"),
            ],
            cwd=ROOT,
        )

    # RQ2 contingency for the paper.
    subprocess.check_call(
        [
            sys.executable,
            str(PAPER / "scripts" / "export_rq2_contingency.py"),
            str(trials),
            "--out",
            str(PAPER / "tables" / "rq2_contingency.tex"),
        ]
    )

    # Claim / artifact audits if available.
    for module, out_name in (
        ("eval.artifact_audit", "artifact_audit.md"),
        ("eval.claim_records", "claim_records.md"),
    ):
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    module,
                    str(run_dir),
                    "--markdown-output",
                    str(run_dir / out_name),
                ],
                cwd=ROOT,
            )
        except subprocess.CalledProcessError as exc:
            print(f"skip {module}: {exc}")

    print(f"Post-processing complete for {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
