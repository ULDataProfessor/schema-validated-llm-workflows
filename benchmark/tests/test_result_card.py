import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.result_card import render_result_card, write_result_card  # noqa: E402


def _write_config(root: Path) -> Path:
    config_path = root / "trials.yaml"
    config_path.write_text(
        """
seeds: [42, 43]
sample_limits:
  extraction: 1
frameworks:
  - pydantic_ai
  - plain_sdk
tasks:
  - extraction
"""
    )
    return config_path


def _write_result_dir(
    root: Path,
    *,
    pilot: bool = False,
    dry_run: bool = False,
    n: int = 2,
) -> Path:
    result_dir = root / "results" / "run"
    result_dir.mkdir(parents=True)
    frameworks = ["dry_run"] if dry_run else ["pydantic_ai", "plain_sdk"]
    seeds = [42] if pilot else [42, 43]
    meta = {
        "run_id": "run",
        "model": "openai:gpt-4o-mini",
        "seed": 42 if pilot else None,
        "seeds": seeds,
        "dry_run": dry_run,
        "pilot": pilot,
        "frameworks": frameworks,
        "tasks": ["extraction"],
        "task_limits": {"extraction": 1},
        "planned_trials": len(frameworks) * n,
        "n_trials": len(frameworks) * n,
        "environment": {
            "python_version": "3.13.0",
            "platform": "test-platform",
            "package_versions": {"pydantic": "2.10.0"},
            "config_hashes": {"config/trials.yaml": "abc123"},
            "git": {"commit": "abc", "dirty": False},
        },
        "provenance": {
            "started_at_utc": "2026-06-12T00:00:00Z",
            "completed_at_utc": "2026-06-12T00:01:00Z",
            "command_argv": ["python", "scripts/run_benchmark.py", "--all"],
        },
    }
    summary = {
        f"{framework}::extraction": {
            "n": n,
            "schema_pass_rate": 1.0,
            "zero_retry_success_rate": 0.5,
            "mean_quality_score": 0.75,
            "quality_metric": "field_macro_accuracy",
            "error_count": 0,
        }
        for framework in frameworks
    }
    (result_dir / "meta.json").write_text(json.dumps(meta))
    (result_dir / "summary.json").write_text(json.dumps(summary))
    (result_dir / "trials.jsonl").write_text("{}\n")
    return result_dir


def test_render_result_card_documents_eligible_live_run(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path)

    card = render_result_card(result_dir, config_path=config_path)

    assert "Eligibility: **eligible**" in card
    assert "Claim scope: descriptive empirical comparison" in card
    assert "This run can support descriptive framework-task comparisons." in card
    assert "Inferential claims require paired-analysis JSON files" in card
    assert "Started at: `2026-06-12T00:00:00Z`" in card
    assert "Completed at: `2026-06-12T00:01:00Z`" in card
    assert "Command: `python scripts/run_benchmark.py --all`" in card
    assert "| Structured extraction | PydanticAI | 2 | 50.0% | 100.0% | 0.750" in card
    assert "No validation problems or warnings were reported." in card


def test_render_result_card_marks_pilot_scope_when_allowed(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)

    card = render_result_card(result_dir, config_path=config_path, allow_pilot=True)

    assert "Eligibility: **pilot-only**" in card
    assert "runner-provider smoke evidence; no framework ranking claim" in card
    assert "Pilot rows can support runner-provider readiness claims only." in card
    assert "Framework ranking requires the complete configured matrix" in card
    assert "Run is pilot=true; tables are for smoke/pilot reporting only" in card


def test_render_result_card_surfaces_validation_problems(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)

    card = render_result_card(result_dir, config_path=config_path)

    assert "Eligibility: **not eligible**" in card
    assert "This run is incomplete or inconsistent with the configured benchmark." in card
    assert "Run is pilot=true; pilot runs are not eligible for final empirical tables" in card


def test_write_result_card_creates_markdown_file(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path)
    output_path = result_dir / "result_card.md"

    report = write_result_card(result_dir, output_path, config_path=config_path)

    assert report.eligible
    assert output_path.exists()
    assert output_path.read_text().startswith("# Benchmark Result Card: run")
