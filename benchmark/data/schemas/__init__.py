"""Shared Pydantic schemas for all benchmark tasks and frameworks."""

from data.schemas.extraction import ComplaintExtraction, InvoiceExtraction
from data.schemas.compliance import PolicyAnswer, Citation
from data.schemas.summarization import ExecutiveSummary
from data.schemas.workflow import WorkflowResult
from data.schemas.sql import SQLGenerationResult

__all__ = [
    "InvoiceExtraction",
    "ComplaintExtraction",
    "ExecutiveSummary",
    "PolicyAnswer",
    "Citation",
    "WorkflowResult",
    "SQLGenerationResult",
]
