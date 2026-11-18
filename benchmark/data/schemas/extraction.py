from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class InvoiceExtraction(BaseModel):
    """Structured fields extracted from a business invoice."""

    vendor_name: str = Field(..., description="Name of the vendor or supplier")
    invoice_number: str
    invoice_date: date
    due_date: date | None = None
    subtotal: Decimal = Field(..., ge=0)
    tax_amount: Decimal = Field(default=Decimal("0"), ge=0)
    total_amount: Decimal = Field(..., ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    line_item_count: int = Field(..., ge=0)


class ComplaintExtraction(BaseModel):
    """Structured fields from a customer complaint."""

    product_or_service: str
    issue_category: str = Field(
        ...,
        description="One of: billing, delivery, quality, service, safety, other",
    )
    severity: str = Field(..., description="One of: low, medium, high, critical")
    requested_resolution: str
    sentiment: str = Field(..., description="One of: negative, neutral, mixed")
