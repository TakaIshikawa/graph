from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_gap_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: str = "Project") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_gap_csv_empty_input_has_header_only():
    assert export_collection_gap_csv([]) == (
        "collection,source_project,gap_type,previous_unit_id,next_unit_id,gap_size,detail\n"
    )


def test_collection_gap_csv_detects_sequence_gaps_from_aliases():
    text = export_collection_gap_csv(
        [
            unit("a", metadata={"collection": "Inbox", "index": 1}),
            unit("b", metadata={"collection": "Inbox", "position": 4}),
        ]
    )

    assert rows(text) == [
        {
            "collection": "Inbox",
            "source_project": "Project",
            "gap_type": "missing_sequence",
            "previous_unit_id": "a",
            "next_unit_id": "b",
            "gap_size": "2",
            "detail": "missing sequence values after 1 before 4",
        }
    ]


def test_collection_gap_csv_detects_large_date_gaps_and_multiple_collection_keys():
    text = export_collection_gap_csv(
        [
            unit("a", metadata={"playlist": "Queue", "date": "2026-01-01"}),
            unit("b", metadata={"playlist": "Queue", "date": "2026-02-15"}),
        ]
    )

    assert rows(text)[0] == {
        "collection": "Queue",
        "source_project": "Project",
        "gap_type": "large_date_gap",
        "previous_unit_id": "a",
        "next_unit_id": "b",
        "gap_size": "45",
        "detail": "2026-01-01 to 2026-02-15",
    }


def test_collection_gap_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "collection-gaps.csv"
    units = [unit("a", metadata={"shelf": "Books", "order": 1}), unit("b", metadata={"shelf": "Books", "order": 3})]

    expected = export_collection_gap_csv(units)
    stats = export_collection_gap_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "gap_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
