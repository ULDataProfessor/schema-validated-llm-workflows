import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.schemas.compliance import PolicyAnswer  # noqa: E402
from data.schemas.sql import SQLGenerationResult  # noqa: E402
from data.schemas.summarization import ExecutiveSummary  # noqa: E402
from data.schemas.workflow import WorkflowResult  # noqa: E402
from frameworks.base import RunResult  # noqa: E402
from frameworks.agno.runner import AgnoRunner  # noqa: E402
from frameworks.crewai.runner import CrewAIRunner  # noqa: E402
from frameworks.dspy.runner import DSPyRunner  # noqa: E402
from frameworks.langgraph.runner import LangGraphRunner, _with_structured_output  # noqa: E402
from frameworks.plain_sdk.runner import PlainSDKRunner  # noqa: E402
from frameworks.pydantic_ai.runner import PydanticAIRunner  # noqa: E402
from tasks.prompts import build_task_user_prompt, get_task_system_prompt  # noqa: E402


SQL_SAMPLE = {
    "question": "What is total sales by region?",
    "schema_ddl": "CREATE TABLE sales (region TEXT, amount REAL);",
    "setup_sql": "CREATE TABLE sales (region TEXT, amount REAL);",
    "gold_sql": "SELECT region, SUM(amount) FROM sales GROUP BY region",
    "gold": {
        "sql": "SELECT region, SUM(amount) FROM sales GROUP BY region",
        "explanation": "Group by region.",
        "tables_used": ["sales"],
    },
}

SUM_SAMPLE = {
    "text": "Q1 revenue was $10M. The main risk was churn.",
    "gold": {
        "headline": "Q1 revenue $10M",
        "revenue_summary": "Revenue was $10M.",
        "key_risks": ["churn"],
        "strategic_recommendations": ["expand renewals"],
        "outlook": "Stable.",
    },
}

POLICY_SAMPLE = {
    "document": "Employees must report incidents within 24 hours.",
    "question": "When must employees report incidents?",
    "gold": {
        "answer": "Within 24 hours.",
        "citations": [{"quote": "within 24 hours"}],
        "confidence": 0.95,
        "unsupported_claims": [],
    },
}

WORKFLOW_SAMPLE = {
    "text": "Please refund my duplicate charge.",
    "gold": {
        "request_classification": "refund",
        "tool_used": "create_refund_case",
        "extracted_entities": {"request_id": "REQ-1"},
        "recommendation": "Open refund case.",
        "requires_human_review": False,
    },
}


def test_task_prompt_builder_uses_task_context():
    sql_prompt = build_task_user_prompt("sql_generation", SQL_SAMPLE)
    assert "CREATE TABLE sales" in sql_prompt
    assert SQL_SAMPLE["question"] in sql_prompt
    assert "read-only SELECT" in sql_prompt

    policy_prompt = build_task_user_prompt("compliance_qa", POLICY_SAMPLE)
    assert POLICY_SAMPLE["document"] in policy_prompt
    assert POLICY_SAMPLE["question"] in policy_prompt
    assert "Policy document" in policy_prompt


def test_pydantic_ai_runner_routes_non_extraction_tasks(monkeypatch):
    calls = []

    async def fake_run_agent(self, *, schema_cls, system_prompt, user_prompt):
        calls.append((schema_cls, system_prompt, user_prompt))
        return RunResult(output=None, schema_valid=False)

    monkeypatch.setattr(PydanticAIRunner, "_run_agent", fake_run_agent)
    runner = PydanticAIRunner()

    asyncio.run(runner.run_sql_generation(SQL_SAMPLE))
    asyncio.run(runner.run_summarization(SUM_SAMPLE))
    asyncio.run(runner.run_compliance_qa(POLICY_SAMPLE))
    asyncio.run(runner.run_multi_step(WORKFLOW_SAMPLE))

    assert [call[0] for call in calls] == [
        SQLGenerationResult,
        ExecutiveSummary,
        PolicyAnswer,
        WorkflowResult,
    ]
    assert get_task_system_prompt("sql_generation") in calls[0][1]
    assert SQL_SAMPLE["question"] in calls[0][2]
    assert POLICY_SAMPLE["question"] in calls[2][2]


def test_plain_sdk_runner_routes_non_extraction_tasks(monkeypatch):
    calls = []

    async def fake_run_structured(self, *, schema_cls, system_prompt, user_prompt):
        calls.append((schema_cls, system_prompt, user_prompt))
        return RunResult(output=None, schema_valid=False)

    monkeypatch.setattr(PlainSDKRunner, "_run_structured", fake_run_structured)
    runner = PlainSDKRunner()

    asyncio.run(runner.run_sql_generation(SQL_SAMPLE))
    asyncio.run(runner.run_summarization(SUM_SAMPLE))
    asyncio.run(runner.run_compliance_qa(POLICY_SAMPLE))
    asyncio.run(runner.run_multi_step(WORKFLOW_SAMPLE))

    assert [call[0] for call in calls] == [
        SQLGenerationResult,
        ExecutiveSummary,
        PolicyAnswer,
        WorkflowResult,
    ]
    assert get_task_system_prompt("multi_step") in calls[3][1]
    assert WORKFLOW_SAMPLE["text"] in calls[3][2]


def test_langgraph_runner_routes_non_extraction_tasks(monkeypatch):
    calls = []

    async def fake_run_graph(self, *, schema_cls, system_prompt, user_prompt):
        calls.append((schema_cls, system_prompt, user_prompt))
        return RunResult(output=None, schema_valid=False)

    monkeypatch.setattr(LangGraphRunner, "_run_graph", fake_run_graph)
    runner = LangGraphRunner()

    asyncio.run(runner.run_sql_generation(SQL_SAMPLE))
    asyncio.run(runner.run_summarization(SUM_SAMPLE))
    asyncio.run(runner.run_compliance_qa(POLICY_SAMPLE))
    asyncio.run(runner.run_multi_step(WORKFLOW_SAMPLE))

    assert [call[0] for call in calls] == [
        SQLGenerationResult,
        ExecutiveSummary,
        PolicyAnswer,
        WorkflowResult,
    ]
    assert get_task_system_prompt("compliance_qa") in calls[2][1]
    assert POLICY_SAMPLE["document"] in calls[2][2]


def test_langgraph_uses_function_calling_structured_output_method():
    class LLM:
        def __init__(self):
            self.calls = []

        def with_structured_output(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "structured"

    llm = LLM()

    assert _with_structured_output(llm, WorkflowResult) == "structured"
    assert llm.calls == [
        (
            (WorkflowResult,),
            {"include_raw": True, "method": "function_calling"},
        )
    ]


def test_agno_runner_routes_non_extraction_tasks(monkeypatch):
    calls = []

    async def fake_run_agent(self, *, schema_cls, system_prompt, user_prompt):
        calls.append((schema_cls, system_prompt, user_prompt))
        return RunResult(output=None, schema_valid=False)

    monkeypatch.setattr(AgnoRunner, "_run_agent", fake_run_agent)
    runner = AgnoRunner()

    asyncio.run(runner.run_sql_generation(SQL_SAMPLE))
    asyncio.run(runner.run_summarization(SUM_SAMPLE))
    asyncio.run(runner.run_compliance_qa(POLICY_SAMPLE))
    asyncio.run(runner.run_multi_step(WORKFLOW_SAMPLE))

    assert [call[0] for call in calls] == [
        SQLGenerationResult,
        ExecutiveSummary,
        PolicyAnswer,
        WorkflowResult,
    ]
    assert get_task_system_prompt("summarization") in calls[1][1]
    assert SUM_SAMPLE["text"] in calls[1][2]


def test_dspy_runner_routes_non_extraction_tasks(monkeypatch):
    calls = []

    async def fake_run_predict(self, *, schema_cls, system_prompt, user_prompt):
        calls.append((schema_cls, system_prompt, user_prompt))
        return RunResult(output=None, schema_valid=False)

    monkeypatch.setattr(DSPyRunner, "_run_predict", fake_run_predict)
    runner = DSPyRunner()

    asyncio.run(runner.run_sql_generation(SQL_SAMPLE))
    asyncio.run(runner.run_summarization(SUM_SAMPLE))
    asyncio.run(runner.run_compliance_qa(POLICY_SAMPLE))
    asyncio.run(runner.run_multi_step(WORKFLOW_SAMPLE))

    assert [call[0] for call in calls] == [
        SQLGenerationResult,
        ExecutiveSummary,
        PolicyAnswer,
        WorkflowResult,
    ]
    assert get_task_system_prompt("sql_generation") in calls[0][1]
    assert SQL_SAMPLE["question"] in calls[0][2]
    assert WORKFLOW_SAMPLE["text"] in calls[3][2]


def test_crewai_runner_routes_non_extraction_tasks(monkeypatch):
    calls = []

    async def fake_run_crew(self, *, schema_cls, system_prompt, user_prompt):
        calls.append((schema_cls, system_prompt, user_prompt))
        return RunResult(output=None, schema_valid=False)

    monkeypatch.setattr(CrewAIRunner, "_run_crew", fake_run_crew)
    runner = CrewAIRunner()

    asyncio.run(runner.run_sql_generation(SQL_SAMPLE))
    asyncio.run(runner.run_summarization(SUM_SAMPLE))
    asyncio.run(runner.run_compliance_qa(POLICY_SAMPLE))
    asyncio.run(runner.run_multi_step(WORKFLOW_SAMPLE))

    assert [call[0] for call in calls] == [
        SQLGenerationResult,
        ExecutiveSummary,
        PolicyAnswer,
        WorkflowResult,
    ]
    assert get_task_system_prompt("compliance_qa") in calls[2][1]
    assert POLICY_SAMPLE["document"] in calls[2][2]
    assert WORKFLOW_SAMPLE["text"] in calls[3][2]
