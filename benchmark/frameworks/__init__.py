"""Framework runner registry."""

from __future__ import annotations

from frameworks.base import DryRunRunner, FrameworkRunner
from frameworks.agno.runner import AgnoRunner
from frameworks.crewai.runner import CrewAIRunner
from frameworks.dspy.runner import DSPyRunner
from frameworks.langgraph.runner import LangGraphRunner
from frameworks.plain_sdk.runner import PlainSDKRunner
from frameworks.pydantic_ai.runner import PydanticAIRunner


def get_runner(framework: str, *, model: str, max_retries: int, dry_run: bool = False) -> FrameworkRunner:
    if dry_run:
        return DryRunRunner()
    runners: dict[str, type[FrameworkRunner]] = {
        "pydantic_ai": PydanticAIRunner,
        "plain_sdk": PlainSDKRunner,
        "langgraph": LangGraphRunner,
        "agno": AgnoRunner,
        "dspy": DSPyRunner,
        "crewai": CrewAIRunner,
    }
    if framework not in runners:
        raise NotImplementedError(
            f"Framework '{framework}' not yet implemented. "
            f"Available: {list(runners.keys())} (+ dry_run)"
        )
    cls = runners[framework]
    if framework == "plain_sdk":
        return cls(model=model, max_retries=max_retries)
    return cls(model=model, max_retries=max_retries)
