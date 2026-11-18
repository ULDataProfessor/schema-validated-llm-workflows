from pydantic import BaseModel, Field


class WorkflowResult(BaseModel):
    """Final output of a multi-step business support workflow."""

    request_classification: str = Field(
        ...,
        description="One of: billing, technical, refund, escalation, general",
    )
    tool_used: str
    extracted_entities: dict[str, str] = Field(default_factory=dict)
    recommendation: str
    requires_human_review: bool = False
