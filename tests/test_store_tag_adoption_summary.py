from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _unit(
    unit_id: str,
    tags: list[str],
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    created_at: datetime | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        tags=tags,
        created_at=created_at or _dt(1),
    )


def test_tag_adoption_summary_empty_store_returns_empty(store: Store):
    assert store.tag_adoption_summary() == []


def test_tag_adoption_summary_counts_sources_content_types_and_timestamps(store: Store):
    store.insert_unit(_unit("a", ["ai", "ops"], created_at=_dt(1)))
    store.insert_unit(
        _unit(
            "b",
            ["ai", "ai"],
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.FINDING,
            created_at=_dt(3),
        )
    )
    store.insert_unit(_unit("c", ["ops"], content_type=ContentType.FINDING, created_at=_dt(2)))

    assert store.tag_adoption_summary() == [
        {
            "tag": "ai",
            "unit_count": 2,
            "first_seen_at": _dt(1).isoformat(),
            "last_seen_at": _dt(3).isoformat(),
            "source_project_counts": {"max": 1, "presence": 1},
            "content_type_counts": {"finding": 1, "insight": 1},
        },
        {
            "tag": "ops",
            "unit_count": 2,
            "first_seen_at": _dt(1).isoformat(),
            "last_seen_at": _dt(2).isoformat(),
            "source_project_counts": {"max": 2},
            "content_type_counts": {"finding": 1, "insight": 1},
        },
    ]


def test_tag_adoption_summary_filters_and_limits(store: Store):
    store.insert_unit(_unit("a", ["team.alpha", "team.beta"], created_at=_dt(1)))
    store.insert_unit(_unit("b", ["team.alpha"], created_at=_dt(2)))
    store.insert_unit(_unit("c", ["team.alpha"], source_project="other", created_at=_dt(3)))
    store.insert_unit(_unit("d", ["personal"], created_at=_dt(4)))

    assert [row["tag"] for row in store.tag_adoption_summary(tag_prefix="team.")] == [
        "team.alpha",
        "team.beta",
    ]
    assert store.tag_adoption_summary(source_project=SourceProject.MAX, min_unit_count=2) == [
        {
            "tag": "team.alpha",
            "unit_count": 2,
            "first_seen_at": _dt(1).isoformat(),
            "last_seen_at": _dt(2).isoformat(),
            "source_project_counts": {"max": 2},
            "content_type_counts": {"insight": 2},
        },
    ]
    assert [row["tag"] for row in store.tag_adoption_summary(limit=1)] == ["team.alpha"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_unit_count": 0}, "min_unit_count must be a positive integer"),
        ({"min_unit_count": True}, "min_unit_count must be a positive integer"),
        ({"limit": 0}, "limit must be a positive integer or None"),
        ({"limit": True}, "limit must be a positive integer or None"),
    ],
)
def test_tag_adoption_summary_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.tag_adoption_summary(**kwargs)
