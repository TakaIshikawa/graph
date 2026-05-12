from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_source_entity_activity_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def dt(day: int) -> datetime:
    return datetime(2026, 5, day, 10, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    entity_type: str = "note",
    title: str | None = None,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> KnowledgeUnit:
    created = created_at or dt(1)
    updated = updated_at or created
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=title or f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        created_at=created,
        ingested_at=created,
        updated_at=updated,
    )


def test_source_entity_activity_groups_windows_tags_and_samples():
    text = export_source_entity_activity_markdown(
        [
            unit("b", tags=["ai", "search"], created_at=dt(2), updated_at=dt(5), title="Second"),
            unit("a", tags=["ai", "AI"], created_at=dt(1), updated_at=dt(3), title="First"),
            unit("c", source_project=SourceProject.PRESENCE, entity_type="", tags=["log"], created_at=dt(4), updated_at=dt(4)),
        ],
        sample_limit=1,
    )

    assert (
        "| max | note | 2 | 2026-05-01T10:00:00+00:00 | 2026-05-05T10:00:00+00:00 | "
        "3 | ai (2); AI (1); search (1) | First |"
    ) in text
    assert "| presence | Unknown | 1 | 2026-05-04T10:00:00+00:00 | 2026-05-04T10:00:00+00:00 | 1 | log (1) | Unit c |" in text


def test_source_entity_activity_filters_and_orders_deterministically():
    units = [
        unit("c", source_project="zeta", entity_type="entry"),
        unit("a", source_project="alpha", entity_type="entry"),
        unit("b", source_project="alpha", entity_type="entry"),
    ]

    first = export_source_entity_activity_markdown(units, min_count=2)
    second = export_source_entity_activity_markdown(reversed(units), min_count=2)

    assert first == second
    assert "| alpha | entry | 2 |" in first
    assert "zeta" not in first


def test_source_entity_activity_writes_same_markdown(tmp_path):
    path = tmp_path / "reports" / "source-activity.md"
    units = [unit("a", tags=["one"])]

    text = export_source_entity_activity_markdown(units, sample_limit=0)
    stats = export_source_entity_activity_markdown(units, path, sample_limit=0)

    assert path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "groups_exported": 1,
        "min_count": 1,
        "sample_limit": 0,
        "bytes_written": path.stat().st_size,
    }
    assert "| _None_ |" in text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_count": 0}, "min_count must be a positive integer"),
        ({"sample_limit": -1}, "sample_limit must be a non-negative integer"),
    ],
)
def test_source_entity_activity_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_source_entity_activity_markdown([], **kwargs)


def test_source_entity_activity_is_importable_from_graph_export():
    from graph.export import export_source_entity_activity_markdown as imported

    assert imported is export_source_entity_activity_markdown
