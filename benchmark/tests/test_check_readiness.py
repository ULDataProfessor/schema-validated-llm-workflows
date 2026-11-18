from scripts.check_readiness import (
    build_report,
    count_jsonl,
    missing_framework_dependencies,
    model_provider,
    selected_limits,
)


def test_model_provider_parses_prefixed_model() -> None:
    assert model_provider("openai:gpt-4o-mini") == "openai"
    assert model_provider("anthropic:claude-sonnet-4-20250514") == "anthropic"
    assert model_provider("gpt-4o-mini") is None


def test_count_jsonl_ignores_blank_lines(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n')

    assert count_jsonl(path) == 2
    assert count_jsonl(tmp_path / "missing.jsonl") == 0


def test_selected_limits_uses_full_or_pilot_config() -> None:
    config = {
        "sample_limits": {"extraction": 100},
        "pilot": {"sample_limits": {"extraction": 10}},
    }

    assert selected_limits(config, pilot=False) == {"extraction": 100}
    assert selected_limits(config, pilot=True) == {"extraction": 10}


def test_missing_framework_dependencies_reports_unknown_import() -> None:
    missing = missing_framework_dependencies(["plain_sdk", "definitely_missing_framework"])

    assert "plain_sdk" not in missing
    assert "definitely_missing_framework" not in missing


def test_live_readiness_checks_framework_dependencies() -> None:
    lines, errors = build_report(pilot=True, dry_run=False, all_seeds=False)

    assert any(line.startswith("Framework dependencies:") for line in lines)
    if errors:
        assert any("missing imports" in error for error in errors)
