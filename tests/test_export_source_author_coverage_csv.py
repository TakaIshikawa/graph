from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_author_coverage_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: SourceProject | str | None, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_author_coverage_empty_input_returns_header_only_csv():
    assert export_source_author_coverage_csv([]) == (
        "source_project,unit_count,units_with_author,author_coverage_percent,"
        "distinct_authors,top_authors\n"
    )


def test_source_author_coverage_groups_by_source_and_unknown():
    text = export_source_author_coverage_csv(
        [
            unit("a", SourceProject.MAX, {"author": "Ada Lovelace"}),
            unit("b", SourceProject.MAX, {"authors": ["Grace Hopper", "Ada Lovelace"]}),
            unit("c", SourceProject.MAX, {"owner": "  "}),
            unit("d", None, {"creator": "Unknown Author"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "units_with_author": "2",
            "author_coverage_percent": "66.67",
            "distinct_authors": "2",
            "top_authors": "Ada Lovelace (2); Grace Hopper (1)",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "units_with_author": "1",
            "author_coverage_percent": "100.00",
            "distinct_authors": "1",
            "top_authors": "Unknown Author (1)",
        },
    ]


def test_source_author_coverage_normalizes_keys_values_and_sorting():
    text = export_source_author_coverage_csv(
        [
            unit("a", "Source B", {"Creators": {"name": "not unpacked"}, "publisher": "Publisher B"}),
            unit("b", "Source A", {"byline": ("Beta\nWriter", "Alpha Writer")}),
            unit("c", "Source A", {"owner": {"Beta Writer", "Alpha Writer"}}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Source A",
            "unit_count": "2",
            "units_with_author": "2",
            "author_coverage_percent": "100.00",
            "distinct_authors": "2",
            "top_authors": "Alpha Writer (2); Beta Writer (2)",
        },
        {
            "source_project": "Source B",
            "unit_count": "1",
            "units_with_author": "1",
            "author_coverage_percent": "100.00",
            "distinct_authors": "2",
            "top_authors": "Publisher B (1); {'name': 'not unpacked'} (1)",
        },
    ]


def test_source_author_coverage_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "authors.csv"
    units = [unit("a", "Source A", {"author": "Ada"})]

    expected = export_source_author_coverage_csv(units)
    stats = export_source_author_coverage_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_source_author_coverage_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", {"author": "Beta"}),
        unit("b", "Source A", {"author": "Alpha"}),
        unit("c", "Source A", {"author": "Beta"}),
    ]

    assert export_source_author_coverage_csv(units) == export_source_author_coverage_csv(reversed(units))
