import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.tables import render_task_table, validate_result_dir, write_tables  # noqa: E402


def _write_result_dir(
    root: Path,
    *,
    dry_run: bool = False,
    pilot: bool = False,
    n: int = 2,
) -> Path:
    result_dir = root / "results" / "run"
    result_dir.mkdir(parents=True)
    frameworks = ["dry_run"] if dry_run else ["pydantic_ai", "plain_sdk"]
    tasks = ["extraction"]
    seeds = [42] if pilot else [42, 43]
    meta = {
        "run_id": "run",
        "model": "openai:gpt-4o-mini",
        "seed": 42 if pilot else None,
        "seeds": seeds,
        "dry_run": dry_run,
        "pilot": pilot,
        "frameworks": frameworks,
        "tasks": tasks,
        "task_limits": {"extraction": 1},
        "n_trials": len(frameworks) * n,
    }
    summary = {
        f"{framework}::extraction": {
            "n": n,
            "schema_pass_rate": 1.0,
            "zero_retry_success_rate": 0.5,
            "mean_quality_score": 0.75,
            "quality_metric": "field_macro_accuracy",
            "mean_retries": 1.0,
            "p95_latency_ms": 1200.0,
            "error_count": 1 if framework == "plain_sdk" else 0,
        }
        for framework in frameworks
    }
    (result_dir / "meta.json").write_text(json.dumps(meta))
    (result_dir / "summary.json").write_text(json.dumps(summary))
    (result_dir / "trials.jsonl").write_text("{}\n")
    return result_dir


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


def test_validate_result_dir_accepts_complete_live_matrix(tmp_path):
    result_dir = _write_result_dir(tmp_path)
    config_path = _write_config(tmp_path)

    report = validate_result_dir(result_dir, config_path=config_path)

    assert report.eligible
    assert report.problems == []


def test_validate_result_dir_rejects_missing_seed(tmp_path):
    result_dir = _write_result_dir(tmp_path, n=1)
    config_path = _write_config(tmp_path)

    report = validate_result_dir(result_dir, config_path=config_path)

    assert not report.eligible
    assert "pydantic_ai::extraction has n=1; expected n=2" in report.problems


def test_validate_result_dir_rejects_pilot_by_default(tmp_path):
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)
    config_path = _write_config(tmp_path)

    report = validate_result_dir(result_dir, config_path=config_path)

    assert not report.eligible
    assert "Run is pilot=true; pilot runs are not eligible for final empirical tables" in report.problems


def test_write_tables_allows_pilot_when_explicit(tmp_path):
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)
    config_path = _write_config(tmp_path)
    output_path = result_dir / "tables.tex"

    report = write_tables(
        result_dir,
        output_path,
        config_path=config_path,
        allow_pilot=True,
    )

    text = output_path.read_text()
    assert report.eligible
    assert "Run is pilot=true; tables are for smoke/pilot reporting only" in report.warnings
    assert "% Eligibility: pilot-only" in text
    assert "% WARNING: Run is pilot=true; tables are for smoke/pilot reporting only" in text


def test_write_tables_includes_generation_comments_and_rows(tmp_path):
    result_dir = _write_result_dir(tmp_path, dry_run=True)
    config_path = _write_config(tmp_path)
    output_path = result_dir / "tables.tex"

    report = write_tables(
        result_dir,
        output_path,
        config_path=config_path,
        allow_dry_run=True,
    )

    text = output_path.read_text()
    assert report.eligible
    assert "% Eligibility: eligible" in text
    assert "Dry run & 2 & 50.0 & 100.0 & 0.75 & 1.00 & 1200.0" in text
    assert r"Quality metric: \texttt{field\_macro\_accuracy}" in text


def test_render_task_table_marks_missing_framework():
    table = render_task_table({}, task="extraction", frameworks=["pydantic_ai"])

    assert "PydanticAI & -- & -- & -- & -- & -- & -- & missing" in table
