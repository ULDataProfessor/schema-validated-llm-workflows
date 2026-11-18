import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.report import aggregate_trials, classify_error, mean_interval, percentile, wilson_interval  # noqa: E402


def test_percentile_interpolates_values():
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([0, 100], 0.95) == 95


def test_wilson_interval_bounds_success_rate():
    low, high = wilson_interval(5, 5)
    assert 0 < low <= 1
    assert high == 1.0


def test_mean_interval_bounds_quality_scores():
    low, high = mean_interval([1.0, 0.0, 1.0], lower_bound=0.0, upper_bound=1.0)
    assert 0.0 <= low < high <= 1.0


def test_classify_error_groups_common_failures():
    assert classify_error("dspy package not installed") == "missing_dependency"
    assert classify_error("BadRequestError: invalid request") == "provider_request"
    assert classify_error("JSON parse failed") == "parse_error"
    assert classify_error(None) is None


def test_aggregate_trials_reports_distribution_and_errors():
    trials = [
        {
            "framework": "plain_sdk",
            "task": "extraction",
            "schema_valid": True,
            "field_f1": 1.0,
            "field_scores": {"vendor": 1.0, "total": 1.0},
            "retries": 0,
            "latency_ms": 100,
            "tokens_in": 20,
            "tokens_out": 10,
            "error": None,
        },
        {
            "framework": "plain_sdk",
            "task": "extraction",
            "schema_valid": False,
            "field_f1": 0.0,
            "field_scores": {"vendor": 0.0, "total": 0.0},
            "retries": 3,
            "latency_ms": 300,
            "tokens_in": 30,
            "tokens_out": 15,
            "error": "ValidationError: missing total",
        },
    ]

    summary = aggregate_trials(trials)["plain_sdk::extraction"]

    assert summary["n"] == 2
    assert summary["schema_pass_rate"] == 0.5
    assert summary["zero_retry_success_rate"] == 0.5
    assert summary["mean_retries"] == 1.5
    assert summary["median_latency_ms"] == 200.0
    assert summary["mean_total_tokens"] == 37.5
    assert summary["mean_quality_score"] == 0.5
    assert summary["quality_metric"] == "field_f1"
    assert summary["field_score_means"] == {"total": 0.5, "vendor": 0.5}
    assert summary["error_count"] == 1
    assert summary["error_type_counts"] == {"schema_violation": 1}


def test_aggregate_trials_prefers_task_aware_quality_score():
    trials = [
        {
            "framework": "pydantic_ai",
            "task": "sql_generation",
            "schema_valid": True,
            "field_f1": 0.6667,
            "quality_score": 0.0,
            "quality_metric": "result_correct",
            "field_scores": {
                "syntax_valid": 1.0,
                "execution_success": 1.0,
                "result_correct": 0.0,
            },
            "retries": 0,
            "latency_ms": 100,
            "tokens_in": 20,
            "tokens_out": 10,
            "error": None,
        }
    ]

    summary = aggregate_trials(trials)["pydantic_ai::sql_generation"]

    assert summary["mean_field_f1"] == 0.6667
    assert summary["mean_quality_score"] == 0.0
    assert summary["quality_metric"] == "result_correct"
