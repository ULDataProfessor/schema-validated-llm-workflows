import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_benchmark import async_main, load_existing_trial_keys, trial_identity  # noqa: E402


def _args(tmp_path: Path, *, resume: bool, limit: int = 1) -> Namespace:
    return Namespace(
        framework="pydantic_ai",
        task="extraction",
        all=True,
        all_seeds=False,
        pilot=True,
        dry_run=True,
        limit=limit,
        seed=None,
        model=None,
        max_retries=None,
        run_id="resume-test",
        results_root=tmp_path / "results",
        resume=resume,
    )


def test_trial_identity_is_stable():
    assert (
        trial_identity(
            framework="pydantic_ai",
            task="extraction",
            sample_id="inv_0001",
            seed=42,
        )
        == "pydantic_ai::extraction::inv_0001::42"
    )


def test_dry_run_resume_skips_existing_trial_rows(tmp_path):
    assert asyncio.run(async_main(_args(tmp_path, resume=False))) == 0

    result_dir = tmp_path / "results" / "resume-test"
    trials_path = result_dir / "trials.jsonl"
    first_lines = trials_path.read_text().splitlines()
    first_meta = json.loads((result_dir / "meta.json").read_text())

    assert len(first_lines) == 5
    assert first_meta["planned_trials"] == 5
    assert first_meta["completed_this_run"] == 5
    assert first_meta["skipped_existing"] == 0
    assert first_meta["environment"]["python_version"]
    assert "pydantic" in first_meta["environment"]["package_versions"]
    assert first_meta["environment"]["config_hashes"]["config/models.yaml"]
    assert first_meta["environment"]["config_hashes"]["config/trials.yaml"]
    assert first_meta["environment"]["config_hashes"]["config/stress.yaml"]
    assert first_meta["provenance"]["started_at_utc"].endswith("Z")
    assert first_meta["provenance"]["completed_at_utc"].endswith("Z")
    assert first_meta["provenance"]["working_directory"]
    assert first_meta["provenance"]["command_argv"]
    assert first_meta["provenance"]["args"]["run_id"] == "resume-test"
    assert first_meta["provenance"]["args"]["results_root"] == str(tmp_path / "results")
    assert len(load_existing_trial_keys(trials_path)) == 5

    assert asyncio.run(async_main(_args(tmp_path, resume=True))) == 0

    second_lines = trials_path.read_text().splitlines()
    second_meta = json.loads((result_dir / "meta.json").read_text())

    assert second_lines == first_lines
    assert second_meta["n_trials"] == 5
    assert second_meta["completed_this_run"] == 0
    assert second_meta["skipped_existing"] == 5


def test_resume_rejects_different_configuration(tmp_path):
    assert asyncio.run(async_main(_args(tmp_path, resume=False, limit=1))) == 0

    try:
        asyncio.run(async_main(_args(tmp_path, resume=True, limit=2)))
    except ValueError as exc:
        assert "Cannot resume result directory with different configuration" in str(exc)
    else:
        raise AssertionError("resume should reject incompatible task limits")
