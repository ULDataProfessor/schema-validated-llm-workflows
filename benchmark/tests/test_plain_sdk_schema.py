import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameworks.plain_sdk.runner import _strict_openai_schema  # noqa: E402


def test_strict_openai_schema_adds_additional_properties_false():
    schema = {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                },
            },
        },
    }

    patched = _strict_openai_schema(schema)

    assert "additionalProperties" not in schema
    assert patched["additionalProperties"] is False
    assert patched["properties"]["customer"]["additionalProperties"] is False
    assert patched["properties"]["lines"]["items"]["additionalProperties"] is False
