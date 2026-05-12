from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_entity_type_heatmap_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str,
    source_entity_type: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_entity_type_heatmap_csv_counts_by_source_and_type():
    text = export_source_entity_type_heatmap_csv(
        [
            unit("b", source_project=SourceProject.PINBOARD, source_entity_type="bookmark"),
            unit("a", source_project=SourceProject.MAX, source_entity_type="note"),
            unit("c", source_project=SourceProject.MAX, source_entity_type="bookmark"),
        ]
    )

    assert text.splitlines()[0] == "source_project,bookmark,note,total"
    assert rows(text) == [
        {"source_project": "max", "bookmark": "1", "note": "1", "total": "2"},
        {"source_project": "pinboard", "bookmark": "1", "note": "0", "total": "1"},
        {"source_project": "__total__", "bookmark": "2", "note": "1", "total": "3"},
    ]


def test_export_source_entity_type_heatmap_csv_is_deterministic_for_reversed_input():
    units = [
        unit("a", source_project="Source B", source_entity_type="zeta"),
        unit("b", source_project="Source A", source_entity_type="alpha"),
        unit("c", source_project="Source A", source_entity_type="zeta"),
    ]

    assert export_source_entity_type_heatmap_csv(units) == export_source_entity_type_heatmap_csv(
        reversed(units)
    )


def test_export_source_entity_type_heatmap_csv_uses_csv_quoting_for_sensitive_values():
    text = export_source_entity_type_heatmap_csv(
        [
            unit("a", source_project="docs, team", source_entity_type="note|draft"),
            unit("b", source_project="docs, team", source_entity_type="quote,clip"),
        ]
    )

    assert text.splitlines()[0] == 'source_project,note|draft,"quote,clip",total'
    assert rows(text)[0] == {
        "source_project": "docs, team",
        "note|draft": "1",
        "quote,clip": "1",
        "total": "2",
    }


def test_export_source_entity_type_heatmap_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "nested" / "heatmap.csv"
    units = [
        unit("a", source_project="Source A", source_entity_type="note"),
        unit("b", source_project="Source B", source_entity_type="note"),
    ]

    expected = export_source_entity_type_heatmap_csv(units)
    stats = export_source_entity_type_heatmap_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "source_project_count": 2,
        "source_entity_type_count": 1,
        "rows_exported": 3,
        "bytes_written": path.stat().st_size,
    }


def test_export_source_entity_type_heatmap_csv_empty_export_has_total_row():
    text = export_source_entity_type_heatmap_csv([])

    assert text == "source_project,total\n__total__,0\n"
