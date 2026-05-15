from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_duplicate_title_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject | str = SourceProject.CSV,
    source_id: str | None = None,
    content_type: ContentType | str = ContentType.INSIGHT,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="item",
        title=title,
        content="",
        content_type=content_type,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_duplicate_title_csv_empty_input_returns_header():
    assert export_unit_duplicate_title_csv([]) == (
        "normalized_title,unit_count,unit_ids,source_projects,source_ids,content_types\n"
    )


def test_unit_duplicate_title_csv_only_emits_shared_normalized_titles():
    text = export_unit_duplicate_title_csv(
        [
            unit("b", "  Project-Plan: v2  ", source_id="doc-2"),
            unit("a", "project plan v2", source_id="doc-1"),
            unit("c", "different title", source_id="doc-3"),
        ]
    )

    assert rows(text) == [
        {
            "normalized_title": "project plan v2",
            "unit_count": "2",
            "unit_ids": "a; b",
            "source_projects": "csv",
            "source_ids": "doc-1; doc-2",
            "content_types": "insight",
        }
    ]


def test_unit_duplicate_title_csv_prefers_canonical_title_metadata():
    text = export_unit_duplicate_title_csv(
        [
            unit("a", "Original A", metadata={"canonical_title": "Canonical / Title"}),
            unit("b", "Original B", metadata={"canonical_title": "canonical title"}),
        ]
    )

    assert rows(text)[0]["normalized_title"] == "canonical title"


def test_unit_duplicate_title_csv_excludes_empty_titles_and_sorts_deterministically():
    units = [
        unit("c", "", metadata={"canonical_title": " "}),
        unit("b", "Zeta Title"),
        unit("a", "zeta-title"),
        unit("d", "Alpha Title"),
        unit("e", "alpha/title"),
    ]

    assert export_unit_duplicate_title_csv(units) == export_unit_duplicate_title_csv(reversed(units))
    assert [row["normalized_title"] for row in rows(export_unit_duplicate_title_csv(units))] == [
        "alpha title",
        "zeta title",
    ]


def test_unit_duplicate_title_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "duplicates.csv"
    units = [unit("a", "Same"), unit("b", "same")]

    expected = export_unit_duplicate_title_csv(units)
    stats = export_unit_duplicate_title_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "duplicate_title_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
