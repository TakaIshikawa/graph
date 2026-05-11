from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_units_to_activity_heatmap_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

BASE_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
    metadata: dict | None = None,
    created_at: datetime = BASE_TIME,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=[],
        metadata=metadata or {},
        created_at=created_at,
        ingested_at=created_at,
        updated_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_activity_heatmap_buckets_by_date_hour_source_and_entity():
    text = export_units_to_activity_heatmap_csv(
        [
            unit("b", source_project=SourceProject.MAX),
            unit("a", source_project=SourceProject.MAX),
            unit("c", source_project=SourceProject.PINBOARD, source_entity_type="bookmark"),
        ]
    )

    assert rows(text) == [
        {
            "date": "2026-05-01",
            "hour": "10",
            "count": "2",
            "source_project": "max",
            "source_entity_type": "note",
        },
        {
            "date": "2026-05-01",
            "hour": "10",
            "count": "1",
            "source_project": "pinboard",
            "source_entity_type": "bookmark",
        },
    ]


def test_activity_heatmap_uses_configured_metadata_date_keys():
    text = export_units_to_activity_heatmap_csv(
        [unit("a", metadata={"source": {"published_at": "2026-06-03T23:04:00Z"}})],
        date_metadata_keys=["source.published_at"],
    )

    assert rows(text)[0] == {
        "date": "2026-06-03",
        "hour": "23",
        "count": "1",
        "source_project": "max",
        "source_entity_type": "note",
    }


def test_activity_heatmap_skips_units_without_parseable_dates():
    invalid = KnowledgeUnit.model_construct(
        id="invalid",
        source_project=SourceProject.MAX,
        source_id="source-invalid",
        source_entity_type="note",
        title="Invalid",
        content="Content",
        content_type=ContentType.INSIGHT,
        tags=[],
        metadata={"published": "not a date"},
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )

    assert export_units_to_activity_heatmap_csv([invalid], date_metadata_keys=["published"]) == (
        "date,hour,count,source_project,source_entity_type\n"
    )


def test_activity_heatmap_writes_to_path(tmp_path):
    path = tmp_path / "reports" / "heatmap.csv"

    stats = export_units_to_activity_heatmap_csv([unit("a")], path)

    assert stats == {
        "path": str(path),
        "rows_written": 1,
        "bytes_written": path.stat().st_size,
    }
    assert rows(path.read_text(encoding="utf-8"))[0]["count"] == "1"


def test_activity_heatmap_is_importable_from_graph_export():
    from graph.export import export_units_to_activity_heatmap_csv as imported

    assert imported is export_units_to_activity_heatmap_csv
