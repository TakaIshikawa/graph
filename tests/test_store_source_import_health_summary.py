from __future__ import annotations

from graph.store.source_import_health_summary import source_import_health_summary


class Unit:
    source_project = "obj"
    source_id = "1"
    content = "body"
    ingested_at = "2024-01-01T00:00:00Z"
    metadata = {"status": "error"}


def test_source_import_health_counts_by_source():
    rows = source_import_health_summary(
        [
            {"source_project": "a", "source_id": "1", "content": "ok", "ingested_at": "bad"},
            {"source_project": "a", "source_id": "1", "content": "", "ingested_at": "2024-02-01T00:00:00Z", "metadata": {"failed": True}},
            {"source_project": "b", "source_id": "1", "content": "ok", "ingested_at": "2024-01-01T00:00:00Z"},
        ]
    )

    assert rows[0] == {
        "source_project": "a",
        "unit_count": 2,
        "latest_ingested_at": "2024-02-01T00:00:00+00:00",
        "missing_source_id_count": 0,
        "duplicate_source_id_count": 2,
        "missing_content_count": 1,
        "error_flag_count": 1,
    }
    assert rows[1]["duplicate_source_id_count"] == 0


def test_source_import_health_supports_objects_and_error_status():
    rows = source_import_health_summary([Unit()])

    assert rows[0]["source_project"] == "obj"
    assert rows[0]["error_flag_count"] == 1
