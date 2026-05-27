from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_sensitive_token_hint_csv import export_units_to_sensitive_token_hint_csv


def test_export_units_to_sensitive_token_hint_csv_redacts_content_values():
    secret = "sk_live_1234567890abcdef"
    token = "abc.def.1234567890abcdef"
    text = export_units_to_sensitive_token_hint_csv(
        [
            {
                "id": "u1",
                "source": "notes",
                "content": f"api_key={secret}\nAuthorization: Bearer {token}\n-----BEGIN PRIVATE KEY-----",
            }
        ]
    )

    assert secret not in text
    assert token not in text
    rows = list(csv.DictReader(StringIO(text)))
    assert [(row["hint_type"], row["line_number"]) for row in rows] == [
        ("secret_assignment", "1"),
        ("bearer_token", "2"),
        ("private_key_marker", "3"),
    ]


def test_export_units_to_sensitive_token_hint_csv_reports_metadata_key_paths():
    secret = "super-secret-value"
    text = export_units_to_sensitive_token_hint_csv(
        [{"id": "u1", "metadata": {"credentials": {"password": secret}, "nested": [{"token": "abc123456"}]}}]
    )

    assert secret not in text
    rows = list(csv.DictReader(StringIO(text)))
    assert [row["location"] for row in rows] == ["metadata.credentials.password", "metadata.nested[0].token"]
    assert {row["hint_type"] for row in rows} == {"metadata_sensitive_key"}
