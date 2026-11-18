#!/usr/bin/env python3
"""Count implementation complexity per framework (LOC, files)."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS_DIR = ROOT / "frameworks"
PYPROJECT = ROOT / "pyproject.toml"

ABSTRACTION_PATTERNS = {
    "agno": [
        r"\bAgent\(",
        r"\bOpenAIResponses\(",
        r"output_schema\s*=",
    ],
    "crewai": [
        r"\bAgent\(",
        r"\bCrew\(",
        r"\bLLM\(",
        r"\bProcess\.",
        r"\bTask\(",
        r"output_pydantic\s*=",
    ],
    "dspy": [
        r"dspy\.InputField\(",
        r"dspy\.JSONAdapter\(",
        r"dspy\.LM\(",
        r"dspy\.OutputField\(",
        r"dspy\.Predict\(",
        r"dspy\.Signature",
    ],
    "langgraph": [
        r"\bStateGraph\(",
        r"\.add_conditional_edges\(",
        r"\.add_edge\(",
        r"\.add_node\(",
        r"\.compile\(",
        r"\.with_structured_output\(",
    ],
    "plain_sdk": [
        r"\bAsyncOpenAI\(",
        r"\.chat\.completions\.create\(",
        r"json_schema",
        r"model_json_schema\(",
        r"response_format\s*=",
    ],
    "pydantic_ai": [
        r"\bAgent\(",
        r"\bModelRetry\b",
        r"\bRunContext\b",
        r"output_type\s*=",
        r"output_validator",
    ],
}


def count_python_loc(path: Path) -> tuple[int, int]:
    """Return (files, loc) for .py files under path."""
    if not path.exists():
        return 0, 0
    files = 0
    loc = 0
    for py in path.rglob("*.py"):
        if py.name.startswith("test_"):
            continue
        files += 1
        loc += sum(1 for line in py.read_text().splitlines() if line.strip())
    return files, loc


def count_abstraction_markers(path: Path, framework: str) -> dict[str, int]:
    """Count transparent framework-specific API markers in implementation files."""
    patterns = {
        pattern: re.compile(pattern)
        for pattern in ABSTRACTION_PATTERNS.get(framework, [])
    }
    counts = {pattern: 0 for pattern in patterns}
    if not path.exists():
        return counts

    for py in path.rglob("*.py"):
        if py.name.startswith("test_"):
            continue
        text = py.read_text()
        for pattern, compiled in patterns.items():
            counts[pattern] += len(compiled.findall(text))
    return counts


def optional_dependency_counts() -> dict[str, dict[str, int | list[str]]]:
    if not PYPROJECT.exists():
        return {}
    data = tomllib.loads(PYPROJECT.read_text())
    optional = data.get("project", {}).get("optional-dependencies", {})
    return {
        name: {
            "dependency_count": len(deps),
            "dependencies": deps,
        }
        for name, deps in optional.items()
    }


def main() -> None:
    report: dict[str, dict] = {}
    dependency_counts = optional_dependency_counts()
    for fw_dir in sorted(FRAMEWORKS_DIR.iterdir()):
        if not fw_dir.is_dir() or fw_dir.name.startswith("_"):
            continue
        if fw_dir.name == "__pycache__":
            continue
        files, loc = count_python_loc(fw_dir)
        marker_counts = count_abstraction_markers(fw_dir, fw_dir.name)
        marker_total = sum(marker_counts.values())
        report[fw_dir.name] = {
            "files": files,
            "loc": loc,
            "abstraction_markers": marker_total,
            "abstraction_marker_counts": marker_counts,
            **dependency_counts.get(fw_dir.name, {"dependency_count": 0, "dependencies": []}),
        }

    total_files = sum(v["files"] for v in report.values())
    total_loc = sum(v["loc"] for v in report.values())
    total_markers = sum(v["abstraction_markers"] for v in report.values())
    report["_total"] = {
        "files": total_files,
        "loc": total_loc,
        "abstraction_markers": total_markers,
    }

    out = ROOT / "results" / "complexity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
