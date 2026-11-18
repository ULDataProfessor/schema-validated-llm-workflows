from pydantic import BaseModel, Field


class ExecutiveSummary(BaseModel):
    """Structured executive summary of a business report."""

    headline: str = Field(..., max_length=200)
    revenue_summary: str
    key_risks: list[str] = Field(..., min_length=1, max_length=5)
    strategic_recommendations: list[str] = Field(..., min_length=1, max_length=5)
    outlook: str = Field(..., description="Forward-looking assessment")
