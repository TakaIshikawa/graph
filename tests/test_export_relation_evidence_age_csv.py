from __future__ import annotations

import csv
from io import StringIO

from graph.export.relation_evidence_age_csv import export_relation_evidence_age_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_evidence_age_csv_emits_dates_and_unknown_bucket():
    text = export_relation_evidence_age_csv(
        [
            {
                "id": "r1",
                "source_id": "u1",
                "target_id": "u2",
                "relation": "supports",
                "metadata": {"evidence": [{"date": "2024-01-01"}, {"metadata": {"observed_at": "2024-02-01T12:00:00Z"}}]},
            },
            {"id": "r2", "source_id": "u3", "target_id": "u4", "relation": "mentions"},
        ]
    )

    by_id = {row["relation_id"]: row for row in rows(text)}
    assert by_id["r1"]["dated_evidence_count"] == "2"
    assert by_id["r1"]["oldest_evidence_date"] == "2024-01-01"
    assert by_id["r1"]["newest_evidence_date"] == "2024-02-01"
    assert by_id["r2"]["age_bucket"] == "unknown"
