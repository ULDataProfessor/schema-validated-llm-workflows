# CrewAI Implementation

`CrewAIRunner` implements the shared benchmark tasks using a single CrewAI
agent, one `Task` per trial, sequential `Crew` execution, and
`output_pydantic` for structured task output. The runner validates the returned
Pydantic or JSON output against the same schemas used by the other framework
conditions.

This is a conservative baseline intended to measure CrewAI's structured-output
task path. Multi-agent variants can be added as separate ablations after the
single-agent matrix is complete.

Implemented tasks:
- structured extraction
- SQL generation
- summarization
- compliance QA
- multi-step workflow
