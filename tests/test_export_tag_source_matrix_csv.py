from __future__ import annotations

import csv
from datetime import datetime, timezone

import pytest

from graph.export import export_tag_source_matrix_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

BASE_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        created_at=BASE_TIME,
        ingested_at=BASE_TIME,
        updated_at=BASE_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def test_tag_source_matrix_counts_tags_by_source_project():
    text = export_tag_source_matrix_csv(
        [
            unit("a", source_project=SourceProject.MAX, tags=["AI", "storage", "storage"]),
            unit("b", source_project=SourceProject.PINBOARD, tags=["AI", " solar\npower "]),
            unit("c", source_project=SourceProject.MAX, tags=["solar power", ""]),
        ]
    )

    assert rows(text) == [
        {"tag": "AI", "max": "1", "pinboard": "1", "total": "2"},
        {"tag": "solar power", "max": "1", "pinboard": "1", "total": "2"},
        {"tag": "storage", "max": "1", "pinboard": "0", "total": "1"},
    ]


def test_tag_source_matrix_sorts_sources_and_tags_deterministically():
    first = export_tag_source_matrix_csv(
        [
            unit("b", source_project="z-source", tags=["beta"]),
            unit("a", source_project="A Source", tags=["alpha"]),
        ]
    )
    second = export_tag_source_matrix_csv(
        [
            unit("a", source_project="A Source", tags=["alpha"]),
            unit("b", source_project="z-source", tags=["beta"]),
        ]
    )

    assert first == second
    assert first.splitlines()[0] == "tag,A Source,z-source,total"
    assert [row["tag"] for row in rows(first)] == ["alpha", "beta"]


def test_tag_source_matrix_filters_by_min_count():
    text = export_tag_source_matrix_csv(
        [
            unit("a", tags=["keep", "drop"]),
            unit("b", tags=["keep"]),
        ],
        min_count=2,
    )

    assert rows(text) == [{"tag": "keep", "max": "2", "total": "2"}]


def test_tag_source_matrix_empty_input_returns_header_only_csv():
    assert export_tag_source_matrix_csv([]) == "tag,total\n"


def test_tag_source_matrix_writes_to_file(tmp_path):
    path = tmp_path / "reports" / "tag-source.csv"

    stats = export_tag_source_matrix_csv([unit("a", tags=["ai"])], path)

    assert stats == {
        "path": str(path),
        "tags_exported": 1,
        "sources_exported": 1,
        "min_count": 1,
        "bytes_written": path.stat().st_size,
    }
    assert path.read_text(encoding="utf-8") == "tag,max,total\nai,1,1\n"


def test_tag_source_matrix_is_importable_from_graph_export():
    from graph.export import export_tag_source_matrix_csv as imported

    assert imported is export_tag_source_matrix_csv


@pytest.mark.parametrize("min_count", [0, -1, True, "2"])
def test_tag_source_matrix_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_tag_source_matrix_csv([], min_count=min_count)
