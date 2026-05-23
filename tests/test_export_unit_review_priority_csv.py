from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_review_priority_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_review_priority_scores_deterministic_signals():
    text = export_unit_review_priority_csv(
        [
            {"id": "clean", "title": "Clean", "content": "Body", "source_project": "src", "tags": ["a", "b"], "updated_at": "2024-01-01"},
            {"id": "risk", "title": "Risk", "content": "", "source_project": "", "tags": [], "updated_at": "2020-01-01"},
        ],
        reference_date="2024-01-01",
    )

    data = rows(text)
    assert data[0] == {
        "unit_id": "risk",
        "title": "Risk",
        "review_priority": "4",
        "reasons": "missing_content; missing_source; low_tag_count; stale_timestamp",
        "missing_signal_count": "4",
    }
    assert data[1]["review_priority"] == "0"


def test_unit_review_priority_counts_unresolved_links_and_path_mode(tmp_path):
    path = tmp_path / "review.csv"
    stats = export_unit_review_priority_csv([{"id": "a", "title": "A", "content": "[[missing]]", "source_id": "s", "tags": ["x"]}], path)

    assert "unresolved_links" in rows(path.read_text(encoding="utf-8"))[0]["reasons"]
    assert stats["rows_exported"] == 1
