from scripts.plan_matrix import (
    build_plan,
    format_plan,
    selected_frameworks,
    selected_sample_limits,
    selected_seeds,
)


TRIALS = {
    "seeds": [42, 43, 44],
    "sample_limits": {
        "extraction": 100,
        "sql_generation": 50,
        "summarization": 50,
        "compliance_qa": 50,
        "multi_step": 50,
    },
    "pilot": {
        "sample_limits": {
            "extraction": 10,
            "sql_generation": 10,
            "summarization": 10,
            "compliance_qa": 10,
            "multi_step": 10,
        }
    },
    "frameworks": ["pydantic_ai", "plain_sdk", "langgraph", "dspy", "crewai", "agno"],
    "tasks": ["extraction", "sql_generation", "summarization", "compliance_qa", "multi_step"],
}


def test_selected_matrix_inputs() -> None:
    assert selected_sample_limits(TRIALS, pilot=False)["extraction"] == 100
    assert selected_sample_limits(TRIALS, pilot=True)["extraction"] == 10
    assert selected_seeds(TRIALS, pilot=True, all_seeds=False) == [42]
    assert selected_seeds(TRIALS, pilot=True, all_seeds=True) == [42, 43, 44]
    assert selected_frameworks(TRIALS, dry_run=False) == TRIALS["frameworks"]
    assert selected_frameworks(TRIALS, dry_run=True) == ["dry_run"]


def test_build_plan_reports_full_live_matrix_size() -> None:
    plans = build_plan(TRIALS)

    by_task = {plan.task: plan for plan in plans}
    assert by_task["extraction"].n_per_framework_task == 300
    assert by_task["sql_generation"].n_per_framework_task == 150
    assert sum(plan.planned_rows for plan in plans) == 5400
    assert 0.05 < by_task["extraction"].max_wilson_half_width < 0.06
    assert 0.07 < by_task["sql_generation"].max_wilson_half_width < 0.09


def test_build_plan_reports_default_pilot_matrix_size() -> None:
    plans = build_plan(TRIALS, pilot=True)

    assert {plan.n_per_framework_task for plan in plans} == {10}
    assert sum(plan.planned_rows for plan in plans) == 300


def test_format_plan_includes_total_rows() -> None:
    plans = build_plan(TRIALS, pilot=True, dry_run=True)
    text = format_plan(
        plans,
        frameworks=["dry_run"],
        seeds=[42],
        pilot=True,
        dry_run=True,
    )

    assert "Mode: dry-run pilot" in text
    assert "extraction" in text
    assert "Total planned rows: 50" in text
