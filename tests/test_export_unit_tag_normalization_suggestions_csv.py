from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_tag_normalization_suggestions_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    source_project: str = "Project A",
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        tags=tags or [],
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_tag_normalization_suggestions_csv_empty_input_has_header_only():
    assert export_unit_tag_normalization_suggestions_csv([]) == (
        "normalized_tag,variant_count,unit_count,variants,suggested_tag,source_projects,unit_ids,confidence\n"
    )


def test_unit_tag_normalization_suggestions_csv_finds_raw_variants():
    text = export_unit_tag_normalization_suggestions_csv(
        [
            unit("b", tags=["Machine-Learning", "ml"]),
            unit("a", tags=["#machine_learning"], source_project="Project B"),
            unit("c", tags=["machine learning"]),
        ]
    )

    assert rows(text) == [
        {
            "normalized_tag": "machine learning",
            "variant_count": "3",
            "unit_count": "3",
            "variants": "#machine_learning; machine learning; Machine-Learning",
            "suggested_tag": "#machine_learning",
            "source_projects": "Project A; Project B",
            "unit_ids": "a; b; c",
            "confidence": "0.77",
        }
    ]


def test_unit_tag_normalization_suggestions_csv_ignores_empty_and_single_variant_groups():
    text = export_unit_tag_normalization_suggestions_csv(
        [
            unit("a", tags=["", "Graph"]),
            unit("b", tags=["Graph"]),
        ]
    )

    assert rows(text) == []


def test_unit_tag_normalization_suggestions_csv_reads_metadata_lists_and_tie_breaks_suggestion():
    text = export_unit_tag_normalization_suggestions_csv(
        [
            unit("a", tags=["rag"], metadata={"tags": ["RAG"], "labels": ["rag"]}),
            unit("b", metadata={"keywords": ["R-A-G"]}),
            unit("c", metadata={"tags": 123}),
        ]
    )

    assert rows(text)[0] == {
        "normalized_tag": "rag",
        "variant_count": "3",
        "unit_count": "2",
        "variants": "R-A-G; RAG; rag",
        "suggested_tag": "rag",
        "source_projects": "Project A",
        "unit_ids": "a; b",
        "confidence": "0.74",
    }


def test_unit_tag_normalization_suggestions_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-tag-normalization-suggestions.csv"
    units = [unit("a", tags=["Data Science"]), unit("b", tags=["data_science"])]

    expected = export_unit_tag_normalization_suggestions_csv(units)
    stats = export_unit_tag_normalization_suggestions_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "suggestion_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
