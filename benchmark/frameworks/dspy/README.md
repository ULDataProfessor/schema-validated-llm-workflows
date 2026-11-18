# DSPy Implementation

`DSPyRunner` implements the shared benchmark tasks using a typed DSPy
signature, `dspy.Predict`, and `dspy.JSONAdapter`. The runner validates the
typed output against the same Pydantic schemas used by the other framework
conditions and records validation failures through the common retry path.

The current runner is an untuned baseline. Optimizer experiments should run on
a development split only and report tuning cost separately per the benchmark
protocol.

Implemented tasks:
- structured extraction
- SQL generation
- summarization
- compliance QA
- multi-step workflow
