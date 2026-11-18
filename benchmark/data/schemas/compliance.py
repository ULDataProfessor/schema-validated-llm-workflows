from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation span from a policy document."""

    document_id: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    quote: str = Field(..., description="Verbatim excerpt from the policy")
    section: str | None = None


class PolicyAnswer(BaseModel):
    """Structured answer to a compliance/policy question."""

    answer: str
    citations: list[Citation] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
