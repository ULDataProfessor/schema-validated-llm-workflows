from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompt_templates"


def get_extraction_prompt(subtype: str = "invoice") -> str:
    filename = "invoice_system.txt" if subtype == "invoice" else "complaint_system.txt"
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    return _DEFAULT_INVOICE if subtype == "invoice" else _DEFAULT_COMPLAINT


_DEFAULT_INVOICE = """You are a business document extraction assistant.
Extract structured invoice fields from the provided document text.
Return only data explicitly present or clearly implied in the document.
Use ISO 8601 dates (YYYY-MM-DD). Monetary amounts as decimal numbers."""

_DEFAULT_COMPLAINT = """You are a customer complaint analysis assistant.
Extract structured fields from the complaint text.
Classify issue_category as one of: billing, delivery, quality, service, safety, other.
Classify severity as one of: low, medium, high, critical.
Classify sentiment as one of: negative, neutral, mixed."""
