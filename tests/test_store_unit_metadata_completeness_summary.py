from __future__ import annotations

from graph.store.unit_metadata_completeness_summary import summarize_unit_metadata_completeness


def test_metadata_completeness_summary_groups_and_counts_present_fields():
    summary = summarize_unit_metadata_completeness(
        [
            {"id": "a", "source_project": "s", "source_entity_type": "note", "title": "A", "metadata": {"tags": ["x"], "language": "en"}},
            {"id": "b", "source_project": "s", "source_entity_type": "note", "title": "", "metadata": {"tags": [], "url": "https://e.test"}},
            {"id": "c", "metadata": {"source": "s", "entity_type": "doc", "attachments": [{"path": "a.pdf"}], "updated_at": "2024-01-01"}},
        ]
    )

    rows = {(row["source"], row["entity_type"]): row for row in summary["rows"]}
    assert rows[("s", "note")]["unit_count"] == 2
    assert rows[("s", "note")]["title_present_count"] == 1
    assert rows[("s", "note")]["tags_coverage_ratio"] == "0.50"
    assert rows[("s", "doc")]["attachments_present_count"] == 1
    assert rows[("s", "doc")]["updated_at_coverage_ratio"] == "1.00"
