from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_tag_overlap_csv import export_source_tag_overlap_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: str,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_tag_overlap_csv_empty_input_has_header_only():
    assert export_source_tag_overlap_csv([]) == (
        "left_source_project,right_source_project,left_tag_count,right_tag_count,"
        "shared_tag_count,jaccard_similarity,shared_tags\n"
    )


def test_source_tag_overlap_csv_emits_one_row_per_unordered_pair():
    text = export_source_tag_overlap_csv(
        [
            unit("b", source_project="B", tags=["AI", "Research"]),
            unit("a", source_project="A", tags=["ai", "Notes"]),
        ]
    )

    assert rows(text) == [
        {
            "left_source_project": "A",
            "right_source_project": "B",
            "left_tag_count": "2",
            "right_tag_count": "2",
            "shared_tag_count": "1",
            "jaccard_similarity": "0.33",
            "shared_tags": "ai",
        }
    ]


def test_source_tag_overlap_csv_reads_metadata_tag_fields():
    text = export_source_tag_overlap_csv(
        [
            unit("a", source_project="A", metadata={"keywords": ["Machine   Learning", "Data"]}),
            unit("b", source_project="B", metadata={"labels": ["machine learning", "Ops"]}),
        ]
    )

    assert rows(text)[0]["shared_tags"] == "Machine Learning"


def test_source_tag_overlap_csv_handles_sources_with_no_tags():
    text = export_source_tag_overlap_csv(
        [
            unit("a", source_project="A"),
            unit("b", source_project="B"),
        ]
    )

    assert rows(text)[0]["jaccard_similarity"] == "0.00"


def test_source_tag_overlap_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-tag-overlap.csv"
    units = [unit("a", source_project="A", tags=["x"]), unit("b", source_project="B", tags=["x"])]

    expected = export_source_tag_overlap_csv(units)
    stats = export_source_tag_overlap_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "source_pair_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
