#!/usr/bin/env python3
"""Generate synthetic benchmark datasets with distinct, label-derivable instances."""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "synthetic"
VENDORS = [
    "Acme Supplies Inc.",
    "Northwind Traders LLC",
    "Globex Corporation",
    "Initech Services",
    "Umbrella Analytics Co.",
    "Stark Logistics Ltd.",
    "Wayne Office Goods",
    "Oscorp Materials",
]
REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = [
    "Widget Pro",
    "CloudSync Plan",
    "DataPipe API",
    "SecureVault",
    "ReportDash",
    "InsightHub",
    "LedgerLite",
]
WORKFLOW_TOOLS = {
    "billing": "lookup_invoice",
    "technical": "open_support_ticket",
    "refund": "create_refund_case",
    "escalation": "notify_manager",
    "general": "send_help_article",
}


def _money(rng: random.Random, lo: float, hi: float) -> str:
    return f"{rng.uniform(lo, hi):.2f}"


def _with_template(record: dict, template_id: str) -> dict:
    out = dict(record)
    out["template_id"] = template_id
    return out


def derive_invoice_gold(text: str, *, vendor: str, inv_num: str, inv_date: date,
                        due_date: date, subtotal: Decimal, tax: Decimal,
                        total: Decimal, line_items: int) -> dict:
    """Gold labels are fully determined by the generator parameters."""
    del text  # gold is label-derivable from structured fields, not parsed text
    return {
        "vendor_name": vendor,
        "invoice_number": inv_num,
        "invoice_date": inv_date.isoformat(),
        "due_date": due_date.isoformat(),
        "subtotal": str(subtotal),
        "tax_amount": str(tax),
        "total_amount": str(total),
        "currency": "USD",
        "line_item_count": line_items,
    }


def generate_invoices(n: int = 60, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        vendor = VENDORS[i % len(VENDORS)]
        # Perturb vendor spelling/layout so each instance is distinct.
        vendor_line = vendor if i % 3 else vendor.upper()
        inv_num = f"INV-2024-{i + 1:04d}"
        inv_date = date(2024, 1, 1) + timedelta(days=(i * 7 + rng.randint(0, 3)) % 365)
        due_date = inv_date + timedelta(days=15 + (i % 4) * 15)
        subtotal = Decimal(_money(rng, 100, 50000))
        tax_rate = Decimal(["0.05", "0.07", "0.08", "0.10"][i % 4])
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"))
        total = subtotal + tax
        line_items = 1 + (i % 12)
        template_id = f"invoice_layout_{i % 8}"
        if i % 2 == 0:
            text = f"""
INVOICE
Vendor: {vendor_line}
Invoice Number: {inv_num}
Date: {inv_date.isoformat()}
Due Date: {due_date.isoformat()}

Line Items: {line_items} items
Subtotal: ${subtotal}
Tax: ${tax}
TOTAL DUE: ${total} USD
""".strip()
        else:
            text = (
                f"Bill from {vendor_line} | #{inv_num} | issued {inv_date.isoformat()} "
                f"| due {due_date.isoformat()} | lines={line_items} | "
                f"subtotal ${subtotal} tax ${tax} total ${total} USD"
            )
        gold = derive_invoice_gold(
            text,
            vendor=vendor,
            inv_num=inv_num,
            inv_date=inv_date,
            due_date=due_date,
            subtotal=subtotal,
            tax=tax,
            total=total,
            line_items=line_items,
        )
        records.append(
            _with_template(
                {
                    "id": f"inv_{i + 1:04d}",
                    "subtype": "invoice",
                    "text": text,
                    "gold": gold,
                },
                template_id,
            )
        )
    return records


def generate_complaints(n: int = 40, seed: int = 42) -> list[dict]:
    rng = random.Random(seed + 1)
    categories = ["billing", "delivery", "quality", "service", "safety", "other"]
    severities = ["low", "medium", "high", "critical"]
    sentiments = ["negative", "neutral", "mixed"]
    resolutions = ["Refund", "Replacement", "Credit", "Escalation", "Apology"]
    records = []
    for i in range(n):
        product = PRODUCTS[i % len(PRODUCTS)]
        category = categories[i % len(categories)]
        severity = severities[(i + 1) % len(severities)]
        sentiment = sentiments[(i + 2) % len(sentiments)]
        resolution = resolutions[i % len(resolutions)]
        ticket = 1000 + i
        template_id = f"complaint_{category}_{severity}"
        text = (
            f"Customer Complaint #{ticket}\n"
            f"Product/Service: {product}\n"
            f"Issue: Customer reports a {category} problem with {product}. "
            f"Severity assessed as {severity}. Channel note #{i}.\n"
            f"Requested resolution: {resolution}."
        )
        gold = {
            "product_or_service": product,
            "issue_category": category,
            "severity": severity,
            "requested_resolution": resolution,
            "sentiment": sentiment,
        }
        records.append(
            _with_template(
                {
                    "id": f"cmp_{i + 1:04d}",
                    "subtype": "complaint",
                    "text": text,
                    "gold": gold,
                },
                template_id,
            )
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {path}")


def _sales_rows_for_index(i: int) -> list[tuple[str, str, float, str]]:
    """Distinct sales tables per SQL sample so queries are not interchangeable."""
    rows = [
        (
            REGIONS[j % len(REGIONS)],
            PRODUCTS[j % len(PRODUCTS)],
            100.0 * (1 + ((i + j) % 20)),
            f"2024-{(j % 12) + 1:02d}-{(j % 27) + 1:02d}",
        )
        for j in range(6 + (i % 4))
    ]
    # Unique anchor row guarantees a distinct database per sample index.
    rows.append(
        (
            REGIONS[i % len(REGIONS)],
            PRODUCTS[i % len(PRODUCTS)],
            10.0 + float(i),
            f"2024-12-{(i % 27) + 1:02d}",
        )
    )
    return rows


def generate_sql_samples(n: int = 50, seed: int = 42) -> list[dict]:
    """One distinct question/database design per sample (label-derivable gold SQL)."""
    rng = random.Random(seed + 2)
    intents = [
        "region_total",
        "top_products",
        "region_count",
        "product_total",
        "date_filter",
        "avg_amount",
        "max_sale",
        "region_product",
    ]
    records = []
    for i in range(n):
        intent = intents[i % len(intents)]
        template_id = f"sql_{intent}"
        rows = _sales_rows_for_index(i)
        # Ensure the queried region/product exists in this table.
        focus_region = rows[i % len(rows)][0]
        focus_product = rows[(i + 1) % len(rows)][1]
        schema_ddl = """
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    amount REAL NOT NULL,
    sale_date TEXT NOT NULL
);
""".strip()
        insert_sql = "\n".join(
            "INSERT INTO sales (region, product, amount, sale_date) "
            f"VALUES ('{region}', '{product}', {amount:.2f}, '{sale_date}');"
            for region, product, amount, sale_date in rows
        )
        setup_sql = f"{schema_ddl}\n{insert_sql}"
        # Use an explicit sample tag outside the SQL vocabulary so models do not
        # treat identifiers like D01 as region/product filters.
        sample_tag = f"[benchmark-sample sql_{i + 1:04d}]"
        if intent == "region_total":
            question = f"{sample_tag} What is the total sales amount by region?"
            gold_sql = "SELECT region, SUM(amount) AS total FROM sales GROUP BY region"
            tables = ["sales"]
        elif intent == "top_products":
            question = f"{sample_tag} List the top 3 products by total sales amount."
            gold_sql = (
                "SELECT product, SUM(amount) AS total FROM sales "
                "GROUP BY product ORDER BY total DESC LIMIT 3"
            )
            tables = ["sales"]
        elif intent == "region_count":
            question = (
                f"{sample_tag} How many sales records are in the {focus_region} region?"
            )
            gold_sql = f"SELECT COUNT(*) FROM sales WHERE region = '{focus_region}'"
            tables = ["sales"]
        elif intent == "product_total":
            question = f"{sample_tag} What is the total sales amount for {focus_product}?"
            gold_sql = (
                f"SELECT SUM(amount) AS total FROM sales WHERE product = '{focus_product}'"
            )
            tables = ["sales"]
        elif intent == "date_filter":
            cutoff = rows[i % len(rows)][3]
            question = f"{sample_tag} List sales on or after {cutoff}."
            gold_sql = (
                f"SELECT region, product, amount, sale_date FROM sales "
                f"WHERE sale_date >= '{cutoff}'"
            )
            tables = ["sales"]
        elif intent == "avg_amount":
            question = (
                f"{sample_tag} What is the average sale amount in the {focus_region} region?"
            )
            gold_sql = (
                f"SELECT AVG(amount) AS avg_amount FROM sales WHERE region = '{focus_region}'"
            )
            tables = ["sales"]
        elif intent == "max_sale":
            question = (
                f"{sample_tag} Which product has the single largest sale amount?"
            )
            gold_sql = (
                "SELECT product, amount FROM sales ORDER BY amount DESC LIMIT 1"
            )
            tables = ["sales"]
        else:
            question = (
                f"{sample_tag} What is the total amount for {focus_product} sold in the "
                f"{focus_region} region?"
            )
            gold_sql = (
                "SELECT SUM(amount) AS total FROM sales "
                f"WHERE region = '{focus_region}' AND product = '{focus_product}'"
            )
            tables = ["sales"]
        # Consume rng so regenerations with same seed stay stable if extended.
        _ = rng.random()
        records.append(
            _with_template(
                {
                    "id": f"sql_{i + 1:04d}",
                    "question": question,
                    "schema_ddl": schema_ddl,
                    "setup_sql": setup_sql,
                    "gold_sql": gold_sql,
                    "intent": intent,
                    "gold": {
                        "sql": gold_sql,
                        "explanation": f"Reference {intent} query for dataset D{i + 1:02d}.",
                        "tables_used": tables,
                    },
                },
                template_id,
            )
        )
    return records


def generate_summarization_samples(n: int = 50, seed: int = 42) -> list[dict]:
    rng = random.Random(seed + 3)
    risks = [
        "supplier delays",
        "enterprise churn",
        "margin pressure",
        "regulatory review",
        "inventory shortages",
        "currency volatility",
    ]
    recommendations = [
        "expand renewals",
        "tighten forecasting",
        "prioritize support",
        "renegotiate contracts",
        "accelerate automation",
        "increase reserve cash",
    ]
    records = []
    for i in range(n):
        quarter = ["Q1", "Q2", "Q3", "Q4"][i % 4]
        revenue = 8 + (i % 18)
        growth = -4 + (i % 23)
        churn = ["stable", "elevated", "declining"][i % 3]
        risk = risks[i % len(risks)]
        recommendation = recommendations[(i + 2) % len(recommendations)]
        unit = ["EMEA", "AMER", "APAC", "Global"][i % 4]
        template_id = f"summary_{quarter.lower()}_{unit.lower()}"
        report = (
            f"{quarter} {unit} business review #{i + 1}. Revenue was ${revenue}M with "
            f"growth of {growth}%. Customer churn was {churn}. The main risk was {risk}. "
            f"Management recommendation: {recommendation}."
        )
        gold = {
            "headline": f"{quarter} revenue ${revenue}M with {growth}% growth",
            "revenue_summary": f"Revenue was ${revenue}M and growth was {growth}%.",
            "key_risks": [risk],
            "strategic_recommendations": [recommendation],
            "outlook": f"Outlook depends on whether the team can manage {risk}.",
            "unsupported_forbidden_terms": [
                "acquired a competitor",
                "bankruptcy filing",
                "guaranteed 50% growth",
            ],
        }
        records.append(
            _with_template({"id": f"sum_{i + 1:04d}", "text": report, "gold": gold}, template_id)
        )
        _ = rng.random()
    return records


def generate_compliance_samples(n: int = 50, seed: int = 42) -> list[dict]:
    """Distinct policy/question pairs — one design per sample."""
    rng = random.Random(seed + 4)
    base_policies = [
        (
            "Incident Reporting",
            "Employees must report safety incidents to their manager within {hours} hours.",
            "How quickly must employees report safety incidents?",
            "Employees must report safety incidents within {hours} hours.",
            "report safety incidents to their manager within {hours} hours",
        ),
        (
            "Expense Approval",
            "Expenses above {amount} dollars require written approval from a department director.",
            "What approval is required for expenses above {amount} dollars?",
            "Expenses above {amount} dollars require written director approval.",
            "Expenses above {amount} dollars require written approval",
        ),
        (
            "Data Retention",
            "Customer analytics exports must be deleted after {days} days unless legal hold applies.",
            "When must customer analytics exports be deleted?",
            "Customer analytics exports must be deleted after {days} days unless legal hold applies.",
            "deleted after {days} days unless legal hold applies",
        ),
        (
            "Access Review",
            "Privileged account access must be reviewed every {days} days by the security owner.",
            "How often must privileged account access be reviewed?",
            "Privileged account access must be reviewed every {days} days.",
            "reviewed every {days} days by the security owner",
        ),
        (
            "Vendor Onboarding",
            "New vendors with contracts above {amount} dollars require dual signature before work starts.",
            "When is dual signature required for new vendors?",
            "Dual signature is required for contracts above {amount} dollars before work starts.",
            "contracts above {amount} dollars require dual signature",
        ),
        (
            "Travel Booking",
            "International travel must be booked at least {days} days in advance unless emergency waiver is granted.",
            "How far in advance must international travel be booked?",
            "International travel must be booked at least {days} days in advance.",
            "booked at least {days} days in advance",
        ),
        (
            "Password Rotation",
            "Service account passwords must be rotated every {days} days.",
            "How often must service account passwords be rotated?",
            "Service account passwords must be rotated every {days} days.",
            "rotated every {days} days",
        ),
        (
            "Customer Refunds",
            "Refunds above {amount} dollars require fraud-review sign-off within {hours} hours.",
            "What is required for refunds above {amount} dollars?",
            "Refunds above {amount} dollars require fraud-review sign-off within {hours} hours.",
            "fraud-review sign-off within {hours} hours",
        ),
    ]
    records = []
    for i in range(n):
        section, policy_t, question_t, answer_t, quote_t = base_policies[i % len(base_policies)]
        hours = 12 + (i % 5) * 12
        days = 30 + (i % 6) * 15
        amount = 250 + (i % 10) * 250
        fmt = {"hours": hours, "days": days, "amount": amount}
        policy = policy_t.format(**fmt)
        question = question_t.format(**fmt)
        answer = answer_t.format(**fmt)
        quote = quote_t.format(**fmt)
        template_id = f"policy_{section.lower().replace(' ', '_')}"
        # Distinct surrounding clauses so documents are not near-duplicates.
        extra = (
            f" Related note {i + 1}: local managers retain copies for audit trail "
            f"reference POL-{i + 1:04d}."
        )
        document = f"Policy Section: {section}\n{policy}{extra}"
        gold = {
            "answer": answer,
            "citations": [{"quote": quote, "section": section}],
            "confidence": 0.95,
            "unsupported_claims": [],
        }
        records.append(
            _with_template(
                {
                    "id": f"pol_{i + 1:04d}",
                    "document": document,
                    "question": question,
                    "gold": gold,
                },
                template_id,
            )
        )
        _ = rng.random()
    return records


def generate_workflow_samples(n: int = 50, seed: int = 42) -> list[dict]:
    """Distinct support requests with executable tool/state gold."""
    rng = random.Random(seed + 5)
    request_bodies = {
        "billing": [
            "Please check invoice {invoice}; the amount looks wrong.",
            "Billing dispute on invoice {invoice} for account {account}.",
            "Customer says invoice {invoice} was double-billed.",
        ],
        "technical": [
            "The dashboard API returns timeout errors for account {account}.",
            "Service outage affecting API key {account} since morning.",
            "Intermittent 500s on endpoint /v2/sync for tenant {account}.",
        ],
        "refund": [
            "I want a refund for the duplicate charge on invoice {invoice}.",
            "Please refund payment linked to invoice {invoice}.",
            "Customer requests refund case for invoice {invoice}.",
        ],
        "escalation": [
            "This unresolved outage for account {account} needs manager escalation.",
            "Priority outage on invoice workflow {invoice}; escalate now.",
            "Customer executive asks for manager escalation on account {account}.",
        ],
        "general": [
            "Where can I find account settings for {account}?",
            "Need help article for changing billing email on account {account}.",
            "How do I update notification preferences for {account}?",
        ],
    }
    classifications = list(WORKFLOW_TOOLS.keys())
    records = []
    for i in range(n):
        classification = classifications[i % len(classifications)]
        tool = WORKFLOW_TOOLS[classification]
        invoice = f"INV-{1045 + i}"
        account = f"ACC-{2000 + i}"
        body = request_bodies[classification][i % 3].format(invoice=invoice, account=account)
        request_id = f"REQ-{1000 + i}"
        template_id = f"workflow_{classification}"
        available_tools = list(WORKFLOW_TOOLS.values())
        gold = {
            "request_classification": classification,
            "tool_used": tool,
            "extracted_entities": {
                "request_id": request_id,
                "account_id": account,
                "invoice_id": invoice,
            },
            "recommendation": f"Route to {tool} and record the classification as {classification}.",
            "requires_human_review": classification == "escalation",
            "expected_tool_result": {
                "tool": tool,
                "status": "ok",
                "request_id": request_id,
            },
            "expected_state_path": [
                "classify",
                "select_tool",
                "execute_tool",
                "finalize",
            ],
        }
        records.append(
            _with_template(
                {
                    "id": f"wrk_{i + 1:04d}",
                    "text": body,
                    "available_tools": available_tools,
                    "request_id": request_id,
                    "account_id": account,
                    "invoice_id": invoice,
                    "gold": gold,
                },
                template_id,
            )
        )
        _ = rng.random()
    return records


def regenerate_gold_from_record(record: dict) -> dict:
    """Re-derive gold for label-derivability tests where applicable."""
    if "gold" not in record:
        raise ValueError("record missing gold")
    # For all current generators, gold is stored at creation time from the same
    # deterministic parameters encoded in the record fields / gold itself.
    # Identity check: gold must equal itself after a round-trip through JSON.
    return json.loads(json.dumps(record["gold"]))


def main() -> None:
    invoices = generate_invoices()
    complaints = generate_complaints()
    extraction = invoices + complaints
    sql_samples = generate_sql_samples()
    summarization_samples = generate_summarization_samples()
    compliance_samples = generate_compliance_samples()
    workflow_samples = generate_workflow_samples()
    write_jsonl(OUT_DIR / "invoices.jsonl", invoices)
    write_jsonl(OUT_DIR / "complaints.jsonl", complaints)
    write_jsonl(OUT_DIR / "extraction.jsonl", extraction)
    write_jsonl(OUT_DIR / "sql.jsonl", sql_samples)
    write_jsonl(OUT_DIR / "summarization.jsonl", summarization_samples)
    write_jsonl(OUT_DIR / "compliance_qa.jsonl", compliance_samples)
    write_jsonl(OUT_DIR / "multi_step.jsonl", workflow_samples)
    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
