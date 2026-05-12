from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_tag_density_csv import export_unit_tag_density_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    tags: list[str] | None = None,
    content: str = "one two three four",
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project A",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Title {unit_id}",
        content=content,
        tags=tags or [],
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_tag_density_csv_empty_input_has_header_only():
    assert export_unit_tag_density_csv([]) == (
        "unit_id,title,source_project,source_entity_type,tag_count,unique_tag_count,"
        "duplicate_tag_count,normalized_tags,content_word_count,tags_per_100_words\n"
    )


def test_unit_tag_density_csv_includes_units_without_tags():
    text = export_unit_tag_density_csv([unit("a", tags=[], content="one two")])

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "Project A",
            "source_entity_type": "note",
            "tag_count": "0",
            "unique_tag_count": "0",
            "duplicate_tag_count": "0",
            "normalized_tags": "",
            "content_word_count": "2",
            "tags_per_100_words": "0.00",
        }
    ]


def test_unit_tag_density_csv_counts_duplicates_after_trim_and_casefold():
    text = export_unit_tag_density_csv(
        [
            unit(
                "a",
                tags=[" Beta ", "alpha", "ALPHA", "beta", "Gamma", ""],
                content="one two three four",
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "Project A",
            "source_entity_type": "note",
            "tag_count": "5",
            "unique_tag_count": "3",
            "duplicate_tag_count": "2",
            "normalized_tags": "ALPHA; Beta; Gamma",
            "content_word_count": "4",
            "tags_per_100_words": "75.00",
        }
    ]


def test_unit_tag_density_csv_zero_word_content_has_zero_density():
    text = export_unit_tag_density_csv([unit("a", tags=["x"], content="")])

    assert rows(text)[0]["content_word_count"] == "0"
    assert rows(text)[0]["tags_per_100_words"] == "0.00"


def test_unit_tag_density_csv_sorts_by_unit_id_then_title():
    units = [
        unit("b", title="Beta", tags=["b"]),
        unit("a", title="Zulu", tags=["z"]),
        unit("a", title="Alpha", tags=["a"]),
    ]

    assert export_unit_tag_density_csv(units) == export_unit_tag_density_csv(reversed(units))
    assert [(row["unit_id"], row["title"]) for row in rows(export_unit_tag_density_csv(units))] == [
        ("a", "Alpha"),
        ("a", "Zulu"),
        ("b", "Beta"),
    ]


def test_unit_tag_density_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-tag-density.csv"
    units = [unit("a", tags=["tag"])]

    expected = export_unit_tag_density_csv(units)
    stats = export_unit_tag_density_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
