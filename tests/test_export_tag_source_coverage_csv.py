from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_tag_source_coverage_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: SourceProject | str | None, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=tags,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tag_source_coverage_empty_input_returns_header_only_csv():
    assert export_tag_source_coverage_csv([]) == (
        "tag,unit_count,distinct_source_count,source_coverage_percent,top_sources\n"
    )


def test_tag_source_coverage_summarizes_tag_sources():
    text = export_tag_source_coverage_csv(
        [
            unit("a", SourceProject.MAX, [" ai ", "research"]),
            unit("b", SourceProject.PINBOARD, ["ai"]),
            unit("c", SourceProject.MAX, ["ai", "ai", ""]),
            unit("d", None, ["research"]),
        ]
    )

    assert rows(text) == [
        {
            "tag": "ai",
            "unit_count": "3",
            "distinct_source_count": "2",
            "source_coverage_percent": "66.67",
            "top_sources": "max (2); pinboard (1)",
        },
        {
            "tag": "research",
            "unit_count": "2",
            "distinct_source_count": "2",
            "source_coverage_percent": "66.67",
            "top_sources": "max (1); Unknown (1)",
        },
    ]


def test_tag_source_coverage_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "tag-sources.csv"
    units = [unit("a", "Source A", ["tag"])]

    expected = export_tag_source_coverage_csv(units)
    stats = export_tag_source_coverage_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "tag_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_tag_source_coverage_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", ["beta"]),
        unit("b", "Source A", ["alpha", "beta"]),
        unit("c", "Source A", ["beta"]),
    ]

    assert export_tag_source_coverage_csv(units) == export_tag_source_coverage_csv(reversed(units))
