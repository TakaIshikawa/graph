from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.kobo_annotations_csv import KoboAnnotationsCsvAdapter
from graph.types.enums import ContentType


def test_kobo_annotations_csv_ingests_highlight_and_note(tmp_path):
    export = tmp_path / "kobo.csv"
    export.write_text(
        "Book Title,Author,Annotation,Note,Chapter,Page,Location,Color,Created Date,Modified Date\n"
        "A Test Book,Jane Writer,Highlighted passage,My marginal note,Chapter 3,42,loc-900,Yellow,2025-01-02T03:04:05Z,2025-01-03T04:05:06Z\n",
        encoding="utf-8",
    )

    result = KoboAnnotationsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "kobo_annotations_csv"
    assert unit.source_id.startswith("kobo_annotations_csv:")
    assert unit.source_entity_type == "annotation"
    assert unit.title == "Kobo annotation: A Test Book"
    assert unit.content_type == ContentType.INSIGHT
    assert "Book: A Test Book" in unit.content
    assert "Author: Jane Writer" in unit.content
    assert "Highlight: Highlighted passage" in unit.content
    assert "Note: My marginal note" in unit.content
    assert unit.metadata["book_title"] == "A Test Book"
    assert unit.metadata["author"] == "Jane Writer"
    assert unit.metadata["highlight"] == "Highlighted passage"
    assert unit.metadata["note"] == "My marginal note"
    assert unit.metadata["chapter"] == "Chapter 3"
    assert unit.metadata["page"] == "42"
    assert unit.metadata["location"] == "loc-900"
    assert unit.metadata["color"] == "Yellow"
    assert unit.metadata["created_date"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["modified_date"] == "2025-01-03T04:05:06+00:00"
    assert unit.tags == ["kobo", "annotation", "yellow"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 3, 4, 5, 6, tzinfo=timezone.utc)


def test_kobo_annotations_csv_stable_ids_and_note_only_rows(tmp_path):
    export = tmp_path / "kobo.csv"
    export.write_text(
        "Title,Note,Chapter,Location,Created\n"
        "Notebook Book,Standalone note,Intro,12,2025-02-01\n",
        encoding="utf-8",
    )

    first = KoboAnnotationsCsvAdapter(path=str(export)).ingest().units[0]
    second = KoboAnnotationsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.metadata["book_title"] == "Notebook Book"
    assert first.metadata["note"] == "Standalone note"
    assert "Note: Standalone note" in first.content
