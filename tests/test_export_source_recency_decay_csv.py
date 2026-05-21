from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_recency_decay_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_recency_decay_csv_summarizes_source_dates():
    text = export_source_recency_decay_csv(
        [
            {"id": "a", "source_project": "A", "metadata": {"published_at": "2026-04-20"}},
            {"id": "b", "source_project": "A", "metadata": {"published_at": "2025-01-01"}},
            {"id": "c", "source_project": "", "metadata": {}},
        ],
        reference_date="2026-05-01",
    )

    result = {row["source_project"]: row for row in rows(text)}
    assert result["A"]["newest_date"] == "2026-04-20"
    assert result["A"]["oldest_date"] == "2025-01-01"
    assert result["A"]["stale_unit_count"] == "1"
    assert result["A"]["fresh_unit_count"] == "1"
    assert result["A"]["decay_band"] == "mixed"
    assert result["Unknown"]["decay_band"] == "no_dates"


def test_export_source_recency_decay_csv_path_mode(tmp_path):
    path = tmp_path / "decay.csv"
    stats = export_source_recency_decay_csv([{"source_project": "A", "metadata": {"date": "2026-05-01"}}], path, reference_date="2026-05-01")

    assert rows(path.read_text(encoding="utf-8"))[0]["decay_band"] == "fresh"
    assert stats["unit_count"] == 1

