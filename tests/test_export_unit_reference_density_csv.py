from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_reference_density_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    content: str = "",
    metadata: object | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=content,
        content_type=ContentType.INSIGHT,
        metadata={} if metadata is None else metadata,
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_reference_density_csv_counts_content_and_metadata_urls():
    text = export_unit_reference_density_csv(
        [
            unit(
                "a",
                content="See https://example.com and [docs](https://docs.example.com).",
                metadata={"url": "http://source.example/a", "links": ["https://extra.example"]},
            ),
            unit("b", content="No references here."),
        ]
    )

    assert text.splitlines()[0] == (
        "source_project,unit_count,total_urls,total_markdown_wiki_references,"
        "units_with_references,average_references_per_unit,unreferenced_unit_count"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "2",
            "total_urls": "4",
            "total_markdown_wiki_references": "1",
            "units_with_references": "1",
            "average_references_per_unit": "2.00",
            "unreferenced_unit_count": "1",
        }
    ]


def test_unit_reference_density_csv_counts_markdown_and_wiki_refs_without_duplicates():
    text = export_unit_reference_density_csv(
        [
            unit(
                "a",
                content=(
                    "[same](https://example.com) [again](https://example.com) "
                    "[[Topic]] and [[Topic]] https://example.com"
                ),
            )
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "1",
            "total_urls": "1",
            "total_markdown_wiki_references": "2",
            "units_with_references": "1",
            "average_references_per_unit": "2.00",
            "unreferenced_unit_count": "0",
        }
    ]


def test_unit_reference_density_csv_groups_missing_source_project_as_unknown():
    text = export_unit_reference_density_csv(
        [
            unit("a", source_project=None, metadata={"source_url": "https://example.com/a"}),
            unit("b", source_project="", content=""),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "unit_count": "2",
            "total_urls": "1",
            "total_markdown_wiki_references": "0",
            "units_with_references": "1",
            "average_references_per_unit": "0.50",
            "unreferenced_unit_count": "1",
        }
    ]


def test_unit_reference_density_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reference-density.csv"
    units = [unit("a", content="https://example.com")]

    expected = export_unit_reference_density_csv(units)
    stats = export_unit_reference_density_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
