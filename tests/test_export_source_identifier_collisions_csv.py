from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_identifier_collisions_csv import export_source_identifier_collisions_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: str = "Project A",
    source_entity_type: str = "note",
    source_id: str = "source-1",
    title: str | None = None,
    content_type: ContentType | str = ContentType.INSIGHT,
    created_at: object = None,
    updated_at: object = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=source_id,
        source_entity_type=source_entity_type,
        title=title or f"Title {unit_id}",
        content="content",
        content_type=content_type,
        metadata={},
        created_at=created_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_identifier_collisions_csv_empty_input_has_header_only():
    assert export_source_identifier_collisions_csv([]) == (
        "source_project,source_entity_type,source_id,unit_count,unit_ids,titles,"
        "content_type_count,first_created_date,last_updated_date\n"
    )


def test_source_identifier_collisions_csv_unique_identifiers_have_header_only():
    text = export_source_identifier_collisions_csv(
        [
            unit("a", source_id="one"),
            unit("b", source_id="two"),
        ]
    )

    assert text == (
        "source_project,source_entity_type,source_id,unit_count,unit_ids,titles,"
        "content_type_count,first_created_date,last_updated_date\n"
    )


def test_source_identifier_collisions_csv_reports_duplicate_source_identifier_in_group():
    text = export_source_identifier_collisions_csv(
        [
            unit(
                "b",
                source_id="same",
                title="Beta",
                content_type=ContentType.FINDING,
                created_at="2024-01-03",
                updated_at="2024-01-05",
            ),
            unit(
                "a",
                source_id="same",
                title="Alpha",
                content_type=ContentType.INSIGHT,
                created_at="2024-01-01T09:00:00Z",
                updated_at="2024-01-10",
            ),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Project A",
            "source_entity_type": "note",
            "source_id": "same",
            "unit_count": "2",
            "unit_ids": "a; b",
            "titles": "Alpha; Beta",
            "content_type_count": "2",
            "first_created_date": "2024-01-01",
            "last_updated_date": "2024-01-10",
        }
    ]


def test_source_identifier_collisions_csv_same_id_in_different_groups_is_not_collision():
    text = export_source_identifier_collisions_csv(
        [
            unit("a", source_project="Project A", source_entity_type="note", source_id="same"),
            unit("b", source_project="Project B", source_entity_type="note", source_id="same"),
            unit("c", source_project="Project A", source_entity_type="task", source_id="same"),
        ]
    )

    assert rows(text) == []


def test_source_identifier_collisions_csv_reports_distinct_titles_even_when_unit_id_matches():
    text = export_source_identifier_collisions_csv(
        [
            unit("a", source_id="same", title="Alpha"),
            unit("a", source_id="same", title="Beta"),
        ]
    )

    assert rows(text)[0]["unit_count"] == "1"
    assert rows(text)[0]["titles"] == "Alpha; Beta"


def test_source_identifier_collisions_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-identifier-collisions.csv"
    units = [
        unit("a", source_id="same", title="Alpha"),
        unit("b", source_id="same", title="Beta"),
    ]

    expected = export_source_identifier_collisions_csv(units)
    stats = export_source_identifier_collisions_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
