from pydantic import BaseModel, Field


class SQLGenerationResult(BaseModel):
    """SQL query generated from a natural language question."""

    sql: str = Field(..., description="Executable SQL query")
    explanation: str = Field(..., description="Brief explanation of the query logic")
    tables_used: list[str] = Field(default_factory=list)
