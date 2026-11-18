import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.count_complexity import count_abstraction_markers, count_python_loc  # noqa: E402


def test_count_python_loc_ignores_blank_lines(tmp_path):
    package = tmp_path / "framework"
    package.mkdir()
    (package / "runner.py").write_text("\nprint('x')\n\n# comment\n")
    (package / "test_runner.py").write_text("print('ignore')\n")

    files, loc = count_python_loc(package)

    assert files == 1
    assert loc == 2


def test_count_abstraction_markers_uses_framework_patterns(tmp_path):
    package = tmp_path / "pydantic_ai"
    package.mkdir()
    (package / "runner.py").write_text(
        "agent = Agent('model', output_type=Invoice)\n"
        "@agent.output_validator\n"
        "def validate(): pass\n"
    )

    counts = count_abstraction_markers(package, "pydantic_ai")

    assert counts[r"\bAgent\("] == 1
    assert counts[r"output_type\s*="] == 1
    assert counts["output_validator"] == 1
