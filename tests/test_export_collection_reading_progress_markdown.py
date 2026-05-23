from __future__ import annotations

from graph.export import export_collection_reading_progress_markdown


def test_collection_reading_progress_empty_input_has_header():
    assert export_collection_reading_progress_markdown([]) == (
        "| collection | total_units | not_started | in_progress | completed | average_progress_percent |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
    )


def test_collection_reading_progress_groups_statuses_and_average_progress():
    text = export_collection_reading_progress_markdown(
        [
            {"id": "a", "metadata": {"collection": "Sci", "status": "completed"}},
            {"id": "b", "metadata": {"collection": "Sci", "pages_read": 25, "total_pages": 100}},
            {"id": "c", "metadata": {"collection": "Sci"}},
        ]
    )

    assert "| Sci | 3 | 1 | 1 | 1 | 41.7 |" in text


def test_collection_reading_progress_handles_multiple_collections_and_path_mode(tmp_path):
    path = tmp_path / "progress.md"
    stats = export_collection_reading_progress_markdown(
        [{"id": "a", "metadata": {"collections": ["A", "B"], "progress": "50%"}}],
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "| A | 1 | 0 | 1 | 0 | 50.0 |" in text
    assert "| B | 1 | 0 | 1 | 0 | 50.0 |" in text
    assert stats["rows_exported"] == 2
