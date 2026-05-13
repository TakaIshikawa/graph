from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_tag_profile_csv import export_source_tag_profile_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MARKDOWN_NOTES,
    source_entity_type: str = "note",
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=unit_id,
        content="",
        tags=tags or [],
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_tag_profile_csv_empty_input_has_header_only():
    assert export_source_tag_profile_csv([]) == (
        "source_project,source_entity_type,unit_count,tagged_unit_count,untagged_unit_count,"
        "distinct_tag_count,total_tag_assignments,average_tags_per_unit,top_tags\n"
    )


def test_source_tag_profile_csv_groups_by_source_project_and_entity_type():
    text = export_source_tag_profile_csv(
        [
            unit("a", source_project=SourceProject.MARKDOWN_NOTES, source_entity_type="note", tags=["ai"]),
            unit("b", source_project=SourceProject.MARKDOWN_NOTES, source_entity_type="task", tags=["todo"]),
            unit("c", source_project=SourceProject.READWISE, source_entity_type="highlight", tags=["ai"]),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "markdown_notes",
            "source_entity_type": "note",
            "unit_count": "1",
            "tagged_unit_count": "1",
            "untagged_unit_count": "0",
            "distinct_tag_count": "1",
            "total_tag_assignments": "1",
            "average_tags_per_unit": "1.00",
            "top_tags": "ai (1)",
        },
        {
            "source_project": "markdown_notes",
            "source_entity_type": "task",
            "unit_count": "1",
            "tagged_unit_count": "1",
            "untagged_unit_count": "0",
            "distinct_tag_count": "1",
            "total_tag_assignments": "1",
            "average_tags_per_unit": "1.00",
            "top_tags": "todo (1)",
        },
        {
            "source_project": "readwise",
            "source_entity_type": "highlight",
            "unit_count": "1",
            "tagged_unit_count": "1",
            "untagged_unit_count": "0",
            "distinct_tag_count": "1",
            "total_tag_assignments": "1",
            "average_tags_per_unit": "1.00",
            "top_tags": "ai (1)",
        },
    ]


def test_source_tag_profile_csv_ignores_blank_tags_and_deduplicates_within_unit():
    text = export_source_tag_profile_csv(
        [
            unit("a", tags=[" ai ", "", "ai", "research", "  "]),
            unit("b", tags=["ai", "planning", "planning"]),
            unit("c", tags=[]),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "markdown_notes",
            "source_entity_type": "note",
            "unit_count": "3",
            "tagged_unit_count": "2",
            "untagged_unit_count": "1",
            "distinct_tag_count": "3",
            "total_tag_assignments": "4",
            "average_tags_per_unit": "1.33",
            "top_tags": "ai (2); planning (1); research (1)",
        }
    ]


def test_source_tag_profile_csv_top_tags_are_deterministic_by_count_then_label():
    units = [
        unit("a", tags=["Beta", "alpha"]),
        unit("b", tags=["gamma", "alpha"]),
        unit("c", tags=["Beta", "delta"]),
    ]

    assert rows(export_source_tag_profile_csv(units))[0]["top_tags"] == "alpha (2); Beta (2); delta (1); gamma (1)"
    assert export_source_tag_profile_csv(list(reversed(units))) == export_source_tag_profile_csv(units)


def test_source_tag_profile_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-tag-profile.csv"
    units = [unit("a", tags=["ai"])]

    expected = export_source_tag_profile_csv(units)
    stats = export_source_tag_profile_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "profile_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
