from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

import pytest

from graph.export import export_tag_timeline_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

BASE_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    tags: list[str] | None = None,
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
        tags=tags or [],
        metadata=metadata or {},
        created_at=created_at,
        ingested_at=created_at,
        updated_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tag_timeline_counts_tags_by_month_source_and_entity():
    text = export_tag_timeline_csv(
        [
            unit("a", tags=["AI", "storage", "storage"]),
            unit("b", tags=["AI", " solar\npower "], source_project=SourceProject.PINBOARD),
            unit("c", tags=["AI"], source_entity_type="bookmark"),
        ]
    )

    assert rows(text) == [
        {
            "period": "2026-05",
            "tag": "AI",
            "count": "1",
            "source_project": "max",
            "source_entity_type": "bookmark",
        },
        {
            "period": "2026-05",
            "tag": "AI",
            "count": "1",
            "source_project": "max",
            "source_entity_type": "note",
        },
        {
            "period": "2026-05",
            "tag": "AI",
            "count": "1",
            "source_project": "pinboard",
            "source_entity_type": "note",
        },
        {
            "period": "2026-05",
            "tag": "solar power",
            "count": "1",
            "source_project": "pinboard",
            "source_entity_type": "note",
        },
        {
            "period": "2026-05",
            "tag": "storage",
            "count": "1",
            "source_project": "max",
            "source_entity_type": "note",
        },
    ]


@pytest.mark.parametrize(
    ("granularity", "expected"),
    [
        ("day", "2026-05-01"),
        ("week", "2026-04-27"),
        ("month", "2026-05"),
        ("year", "2026"),
    ],
)
def test_tag_timeline_supports_granularity_labels(granularity: str, expected: str):
    text = export_tag_timeline_csv([unit("a", tags=["ai"])], granularity=granularity)

    assert rows(text)[0]["period"] == expected


def test_tag_timeline_uses_configured_metadata_date_keys():
    text = export_tag_timeline_csv(
        [unit("a", tags=["ai"], metadata={"source": {"published_at": "2026-06-03"}})],
        granularity="day",
        date_metadata_keys=["source.published_at"],
    )

    assert rows(text)[0]["period"] == "2026-06-03"


def test_tag_timeline_skips_units_without_parseable_dates():
    invalid = KnowledgeUnit.model_construct(
        id="invalid",
        source_project=SourceProject.MAX,
        source_id="source-invalid",
        source_entity_type="note",
        title="Invalid",
        content="Content",
        content_type=ContentType.INSIGHT,
        tags=["ai"],
        metadata={"published": "not a date"},
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )

    assert export_tag_timeline_csv([invalid], date_metadata_keys=["published"]) == (
        "period,tag,count,source_project,source_entity_type\n"
    )


def test_tag_timeline_invalid_granularity_raises_value_error():
    with pytest.raises(ValueError, match="granularity must be one of"):
        export_tag_timeline_csv([], granularity="quarter")


def test_tag_timeline_writes_to_path(tmp_path):
    path = tmp_path / "reports" / "tags.csv"

    stats = export_tag_timeline_csv([unit("a", tags=["ai"])], path)

    assert stats == {
        "path": str(path),
        "rows_written": 1,
        "granularity": "month",
        "bytes_written": path.stat().st_size,
    }
    assert rows(path.read_text(encoding="utf-8"))[0]["tag"] == "ai"


def test_tag_timeline_is_importable_from_graph_export():
    from graph.export import export_tag_timeline_csv as imported

    assert imported is export_tag_timeline_csv
