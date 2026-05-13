from __future__ import annotations

import csv
from io import StringIO

from graph.export.collection_chronology_csv import export_collection_chronology_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: str = "A") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=[],
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_chronology_csv_empty_input_has_header_only():
    assert export_collection_chronology_csv([]) == (
        "collection,unit_count,source_projects,first_seen_date,last_seen_date,span_days,dated_unit_count\n"
    )


def test_collection_chronology_csv_groups_collection_dates():
    text = export_collection_chronology_csv(
        [
            unit("b", metadata={"collection": "Inbox", "date": "2024-01-10"}, source_project="B"),
            unit("a", metadata={"collection": "Inbox", "date": "2024-01-01T09:00:00Z"}, source_project="A"),
        ]
    )

    assert rows(text) == [
        {
            "collection": "Inbox",
            "unit_count": "2",
            "source_projects": "A; B",
            "first_seen_date": "2024-01-01",
            "last_seen_date": "2024-01-10",
            "span_days": "9",
            "dated_unit_count": "2",
        }
    ]


def test_collection_chronology_csv_units_with_multiple_collections_contribute_to_each():
    text = export_collection_chronology_csv([unit("a", metadata={"collection": ["Inbox", "Archive"], "date": "2024-01-01"})])

    assert [row["collection"] for row in rows(text)] == ["Archive", "Inbox"]


def test_collection_chronology_csv_collections_without_dates_emit_blank_date_fields():
    text = export_collection_chronology_csv([unit("a", metadata={"board_name": "Ideas", "date": "bad"})])

    assert rows(text)[0] == {
        "collection": "Ideas",
        "unit_count": "1",
        "source_projects": "A",
        "first_seen_date": "",
        "last_seen_date": "",
        "span_days": "",
        "dated_unit_count": "0",
    }


def test_collection_chronology_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "collection-chronology.csv"
    units = [unit("a", metadata={"list_name": "Reading", "date": "2024-01-01"})]

    expected = export_collection_chronology_csv(units)
    stats = export_collection_chronology_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "collection_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
