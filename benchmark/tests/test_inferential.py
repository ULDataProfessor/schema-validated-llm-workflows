import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.inferential import (  # noqa: E402
    binary_pair_counts,
    bootstrap_ci,
    metric_value,
    pair_clusters,
    paired_differences,
    paired_records,
    paired_summary,
    write_paired_summary,
)


def trial(
    framework: str,
    sample_id: str,
    *,
    task: str = "extraction",
    seed: int = 1,
    schema_valid: bool = True,
    retries: int = 0,
    quality_score: float = 1.0,
    latency_ms: float = 100.0,
) -> dict:
    return {
        "framework": framework,
        "task": task,
        "sample_id": sample_id,
        "seed": seed,
        "schema_valid": schema_valid,
        "field_f1": quality_score,
        "quality_score": quality_score,
        "field_scores": {"answer": quality_score},
        "retries": retries,
        "latency_ms": latency_ms,
        "tokens_in": 20,
        "tokens_out": 5,
    }


def test_paired_differences_use_only_matched_framework_task_keys():
    trials = [
        trial("pydantic_ai", "a", quality_score=0.9),
        trial("plain_sdk", "a", quality_score=0.7),
        trial("pydantic_ai", "b", quality_score=0.4),
        trial("plain_sdk", "b", quality_score=0.6),
        trial("pydantic_ai", "unmatched_a", quality_score=1.0),
        trial("plain_sdk", "unmatched_b", quality_score=1.0),
        trial("pydantic_ai", "a", task="sql_generation", quality_score=0.0),
        trial("plain_sdk", "a", task="sql_generation", quality_score=1.0),
    ]

    differences = paired_differences(
        trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="quality_score",
    )

    assert [round(value, 4) for value in differences] == [0.2, -0.2]


def test_metric_value_supports_schema_zero_retry_tokens_and_field_scores():
    record = trial(
        "pydantic_ai",
        "a",
        schema_valid=True,
        retries=0,
        quality_score=0.75,
    )

    assert metric_value(record, "schema_valid") == 1.0
    assert metric_value(record, "zero_retry_success") == 1.0
    assert metric_value(record, "total_tokens") == 25.0
    assert metric_value(record, "field:answer") == 0.75

    invalid_retry = trial("plain_sdk", "a", schema_valid=True, retries=1)
    assert metric_value(invalid_retry, "zero_retry_success") == 0.0


def test_paired_summary_counts_direction_for_quality_and_latency():
    quality_trials = [
        trial("pydantic_ai", "a", quality_score=0.9),
        trial("plain_sdk", "a", quality_score=0.7),
        trial("pydantic_ai", "b", quality_score=0.4),
        trial("plain_sdk", "b", quality_score=0.6),
    ]

    quality = paired_summary(
        quality_trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="quality_score",
        bootstrap_samples=50,
        seed=2,
    )

    assert quality["n_pairs"] == 2
    assert quality["mean_a_minus_b"] == 0.0
    assert quality["direction"] == "higher_is_better"
    assert quality["a_better"] == 1
    assert quality["b_better"] == 1

    latency_trials = [
        trial("pydantic_ai", "a", latency_ms=100),
        trial("plain_sdk", "a", latency_ms=150),
    ]
    latency = paired_summary(
        latency_trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="latency_ms",
    )

    assert latency["mean_a_minus_b"] == -50.0
    assert latency["direction"] == "lower_is_better"
    assert latency["a_better"] == 1


def test_binary_pair_counts_report_matched_success_failures():
    pairs = [
        (("extraction", "a", "1"), 1.0, 0.0),
        (("extraction", "b", "1"), 0.0, 1.0),
        (("extraction", "c", "1"), 1.0, 1.0),
        (("extraction", "d", "1"), 0.0, 0.0),
    ]

    assert binary_pair_counts(pairs) == {
        "a_success_b_failure": 1,
        "b_success_a_failure": 1,
        "both_success": 1,
        "both_failure": 1,
        "discordant_pairs": 2,
    }


def test_pair_clusters_group_repeated_seeds_by_task_and_sample():
    trials = [
        trial("pydantic_ai", "a", seed=1, quality_score=1.0),
        trial("plain_sdk", "a", seed=1, quality_score=0.0),
        trial("pydantic_ai", "a", seed=2, quality_score=1.0),
        trial("plain_sdk", "a", seed=2, quality_score=0.0),
        trial("pydantic_ai", "b", seed=1, quality_score=0.0),
        trial("plain_sdk", "b", seed=1, quality_score=1.0),
    ]
    pairs = paired_records(
        trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="quality_score",
    )

    assert pair_clusters(pairs) == [
        "extraction::a",
        "extraction::a",
        "extraction::b",
    ]


def test_paired_summary_defaults_to_task_sample_cluster_bootstrap():
    trials = [
        trial("pydantic_ai", "a", seed=1, quality_score=1.0),
        trial("plain_sdk", "a", seed=1, quality_score=0.0),
        trial("pydantic_ai", "a", seed=2, quality_score=1.0),
        trial("plain_sdk", "a", seed=2, quality_score=0.0),
        trial("pydantic_ai", "b", seed=1, quality_score=0.0),
        trial("plain_sdk", "b", seed=1, quality_score=1.0),
    ]

    summary = paired_summary(
        trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="quality_score",
        bootstrap_samples=20,
        seed=2,
    )
    row_summary = paired_summary(
        trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="quality_score",
        bootstrap_samples=20,
        seed=2,
        cluster_bootstrap=False,
    )

    assert summary["n_pairs"] == 3
    assert summary["n_clusters"] == 2
    assert summary["bootstrap_cluster"] == "task_sample"
    assert row_summary["n_clusters"] == 3
    assert row_summary["bootstrap_cluster"] == "trial_row"


def test_paired_summary_includes_binary_pair_counts_for_schema_validity():
    trials = [
        trial("pydantic_ai", "a", schema_valid=True),
        trial("plain_sdk", "a", schema_valid=False),
        trial("pydantic_ai", "b", schema_valid=False),
        trial("plain_sdk", "b", schema_valid=True),
        trial("pydantic_ai", "c", schema_valid=True),
        trial("plain_sdk", "c", schema_valid=True),
        trial("pydantic_ai", "d", schema_valid=False),
        trial("plain_sdk", "d", schema_valid=False),
    ]

    summary = paired_summary(
        trials,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        metric="schema_valid",
        bootstrap_samples=50,
        seed=2,
    )

    assert summary["a_success_b_failure"] == 1
    assert summary["b_success_a_failure"] == 1
    assert summary["both_success"] == 1
    assert summary["both_failure"] == 1
    assert summary["discordant_pairs"] == 2


def test_bootstrap_ci_is_deterministic_and_handles_small_samples():
    assert bootstrap_ci([], n_resamples=10) == [0.0, 0.0]
    assert bootstrap_ci([0.25], n_resamples=10) == [0.25, 0.25]

    first = bootstrap_ci([0.2, -0.1, 0.3], n_resamples=100, seed=3)
    second = bootstrap_ci([0.2, -0.1, 0.3], n_resamples=100, seed=3)

    assert first == second
    assert first[0] <= 0.1334 <= first[1]


def test_write_paired_summary_writes_json(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    output_path = tmp_path / "paired.json"
    records = [
        trial("pydantic_ai", "a", quality_score=1.0),
        trial("plain_sdk", "a", quality_score=0.5),
    ]
    trials_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    summary = write_paired_summary(
        trials_path,
        output_path,
        framework_a="pydantic_ai",
        framework_b="plain_sdk",
        task="extraction",
        bootstrap_samples=20,
    )

    written = json.loads(output_path.read_text())
    assert summary["mean_a_minus_b"] == 0.5
    assert written["n_pairs"] == 1
    assert written["ci95_mean_a_minus_b"] == [0.5, 0.5]
