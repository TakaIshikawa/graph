from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_ingest_latency_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_ingest_latency_computes_hours_and_bucket():
    result = rows(
        export_units_to_ingest_latency_csv(
            [
                {
                    "id": "u1",
                    "source_project": "demo",
                    "source_entity_type": "post",
                    "metadata": {"source_created_at": "2024-01-01T00:00:00Z"},
                    "ingested_at": "2024-01-02T12:00:00Z",
                }
            ]
        )
    )[0]

    assert result["unit_id"] == "u1"
    assert result["source"] == "demo"
    assert result["entity_type"] == "post"
    assert result["latency_hours"] == "36.00"
    assert result["latency_bucket"] == "1d_7d"


def test_ingest_latency_negative_bucket_is_deterministic():
    result = rows(
        export_units_to_ingest_latency_csv(
            [{"id": "u1", "metadata": {"created_at": "2024-01-02T00:00:00Z"}, "ingested_at": "2024-01-01T23:00:00Z"}]
        )
    )[0]

    assert result["latency_hours"] == "-1.00"
    assert result["latency_bucket"] == "negative"


def test_ingest_latency_skips_missing_and_invalid_dates():
    assert export_units_to_ingest_latency_csv(
        [
            {"id": "bad", "metadata": {"created_at": "nope"}, "ingested_at": "2024-01-01T00:00:00Z"},
            {"id": "missing", "metadata": {"created_at": "2024-01-01T00:00:00Z"}},
        ]
    ) == "unit_id,source,entity_type,source_created_at,ingested_at,latency_hours,latency_bucket\n"
