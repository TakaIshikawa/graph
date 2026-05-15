from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_title_similarity_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, *, metadata: dict | None = None, source_entity_type: str = "item") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title,
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_title_similarity_csv_empty_input_returns_header():
    assert export_source_title_similarity_csv([]) == (
        "normalized_title,title_variants,unit_count,source_observation_count,source_types,unit_ids\n"
    )


def test_export_source_title_similarity_csv_groups_normalized_duplicate_titles():
    text = export_source_title_similarity_csv(
        [
            unit("a", "Example: Title!"),
            unit("b", "example title"),
            unit("c", "Different"),
        ]
    )

    assert rows(text) == [
        {
            "normalized_title": "example title",
            "title_variants": "example title; Example: Title!",
            "unit_count": "2",
            "source_observation_count": "2",
            "source_types": "item",
            "unit_ids": "a; b",
        }
    ]


def test_export_source_title_similarity_csv_counts_source_observations():
    text = export_source_title_similarity_csv(
        [
            unit(
                "a",
                "Not duplicate",
                metadata={"sources": [{"title": "Shared Title", "type": "web"}, {"name": "shared-title", "type": "note"}]},
            )
        ]
    )

    row = rows(text)[0]
    assert row["normalized_title"] == "shared title"
    assert row["unit_count"] == "1"
    assert row["source_observation_count"] == "2"
    assert row["source_types"] == "note; web"


def test_export_source_title_similarity_csv_omits_single_observation_groups():
    assert rows(export_source_title_similarity_csv([unit("a", "Only One")])) == []


def test_export_source_title_similarity_csv_path_mode(tmp_path):
    path = tmp_path / "titles.csv"
    stats = export_source_title_similarity_csv([unit("a", "Same"), unit("b", "same")], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["normalized_title"] == "same"
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
