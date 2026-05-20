from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_tag_entropy_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: str = "Project",
    tags: object = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[] if tags is None else tags,
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_tag_entropy_csv_empty_input_has_header_only():
    assert export_source_tag_entropy_csv([]) == (
        "source,unit_count,tagged_unit_count,unique_tag_count,tag_entropy,top_tag,top_tag_share\n"
    )


def test_source_tag_entropy_csv_handles_tag_shapes_and_counts_concentration():
    text = export_source_tag_entropy_csv(
        [
            unit("a", tags=["AI", "Graph"], metadata={"tags": "ai, Search"}),
            unit("b", tags=("ai",), metadata={"tag": {"Graph"}}),
            unit("c", tags=[]),
        ]
    )

    assert rows(text) == [
        {
            "source": "Project",
            "unit_count": "3",
            "tagged_unit_count": "2",
            "unique_tag_count": "3",
            "tag_entropy": "1.46",
            "top_tag": "AI",
            "top_tag_share": "0.50",
        }
    ]


def test_source_tag_entropy_csv_entropy_zero_for_single_unique_tag():
    text = export_source_tag_entropy_csv([unit("a", source_project="Solo", tags="Only"), unit("b", source_project="Solo", tags="only")])

    assert rows(text)[0]["tag_entropy"] == "0.00"
    assert rows(text)[0]["top_tag_share"] == "1.00"


def test_source_tag_entropy_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "entropy.csv"
    units = [unit("a", tags=["one"])]

    expected = export_source_tag_entropy_csv(units)
    stats = export_source_tag_entropy_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
