from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_duplicate_titles_csv import export_unit_duplicate_titles_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str,
    source_project: str = "Project A",
    source_entity_type: str = "note",
    content_type: ContentType | str = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=title,
        content="content",
        content_type=content_type,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_duplicate_titles_csv_empty_input_has_header_only():
    assert export_unit_duplicate_titles_csv([]) == (
        "normalized_title,display_title,duplicate_count,unit_ids,source_projects,"
        "source_entity_types,content_types\n"
    )


def test_unit_duplicate_titles_csv_normalizes_case_and_whitespace():
    text = export_unit_duplicate_titles_csv(
        [
            unit("b", title="  Duplicate\tTitle ", source_project="Project B"),
            unit("a", title="duplicate  title", source_project="Project A", content_type=ContentType.FINDING),
            unit("c", title="Unique"),
        ]
    )

    assert rows(text) == [
        {
            "normalized_title": "duplicate title",
            "display_title": "Duplicate Title",
            "duplicate_count": "2",
            "unit_ids": "a; b",
            "source_projects": "Project A; Project B",
            "source_entity_types": "note",
            "content_types": "finding; insight",
        }
    ]


def test_unit_duplicate_titles_csv_only_emits_groups_with_two_or_more_units():
    text = export_unit_duplicate_titles_csv(
        [
            unit("a", title="One"),
            unit("b", title="Two"),
        ]
    )

    assert rows(text) == []


def test_unit_duplicate_titles_csv_deduplicates_and_sorts_grouped_values():
    text = export_unit_duplicate_titles_csv(
        [
            unit("b", title="Same", source_project="B", source_entity_type="task", content_type="zeta"),
            unit("a", title=" same ", source_project="A", source_entity_type="note", content_type="alpha"),
            unit("a", title="SAME", source_project="A", source_entity_type="note", content_type="alpha"),
        ]
    )

    assert rows(text)[0] == {
        "normalized_title": "same",
        "display_title": "Same",
        "duplicate_count": "3",
        "unit_ids": "a; b",
        "source_projects": "A; B",
        "source_entity_types": "note; task",
        "content_types": "alpha; zeta",
    }


def test_unit_duplicate_titles_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-duplicate-titles.csv"
    units = [
        unit("a", title="Same"),
        unit("b", title="same"),
    ]

    expected = export_unit_duplicate_titles_csv(units)
    stats = export_unit_duplicate_titles_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "duplicate_group_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
