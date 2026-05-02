from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _unit(
    source_id: str,
    title: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    tags: list[str] | None = None,
    created_at: str = "2024-01-01T00:00:00+00:00",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=source_project,
        source_id=source_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=content_type,
        tags=tags or [],
        created_at=created_at,
        updated_at=created_at,
    )


def _set_unit_times(
    store: Store,
    unit_id: str,
    *,
    created_at: str,
    ingested_at: str | None = None,
    updated_at: str | None = None,
) -> None:
    store.conn.execute(
        """UPDATE knowledge_units
           SET created_at = ?, ingested_at = ?, updated_at = ?
           WHERE id = ?""",
        (created_at, ingested_at or created_at, updated_at or created_at, unit_id),
    )
    store.conn.commit()


def test_collection_activity_summary_returns_clear_missing_collection_result(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        summary = store.collection_activity_summary("missing")
    finally:
        store.close()

    assert summary == {
        "collection": "missing",
        "buckets": [],
        "source_project_counts": {},
        "content_type_counts": {},
        "tag_counts": {},
        "first_seen_at": None,
        "last_seen_at": None,
        "error": "collection_not_found",
        "message": "Collection not found: missing",
    }


def test_collection_activity_summary_handles_empty_collections(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.create_collection("empty", metadata={"owner": "ops"})

        summary = store.collection_activity_summary("empty")
    finally:
        store.close()

    assert summary["collection"]["name"] == "empty"
    assert summary["collection"]["metadata"] == {"owner": "ops"}
    assert summary["collection"]["unit_count"] == 0
    assert summary["buckets"] == []
    assert summary["source_project_counts"] == {}
    assert summary["content_type_counts"] == {}
    assert summary["tag_counts"] == {}
    assert summary["first_seen_at"] is None
    assert summary["last_seen_at"] is None


def test_collection_activity_summary_counts_members_only_by_source_type_and_tag(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        jan_max = store.insert_unit(
            _unit(
                "jan-max",
                "January max",
                source_project=SourceProject.MAX,
                content_type=ContentType.INSIGHT,
                tags=["solar", "storage"],
                created_at="2024-01-10T09:00:00+00:00",
            )
        )
        jan_me = store.insert_unit(
            _unit(
                "jan-me",
                "January me",
                source_project=SourceProject.ME,
                content_type=ContentType.FINDING,
                tags=["solar", "grid"],
                created_at="2024-01-20T09:00:00+00:00",
            )
        )
        feb_max = store.insert_unit(
            _unit(
                "feb-max",
                "February max",
                source_project=SourceProject.MAX,
                content_type=ContentType.FINDING,
                tags=["grid"],
                created_at="2024-02-01T09:00:00+00:00",
            )
        )
        nonmember = store.insert_unit(
            _unit(
                "nonmember",
                "Nonmember",
                source_project=SourceProject.PRESENCE,
                content_type=ContentType.ARTIFACT,
                tags=["outside"],
                created_at="2024-02-03T09:00:00+00:00",
            )
        )

        store.create_collection("review")
        for unit in (jan_max, jan_me, feb_max):
            store.add_unit_to_collection("review", unit.id)

        summary = store.collection_activity_summary("review")
    finally:
        store.close()

    assert nonmember.id
    assert summary["buckets"] == [
        {"bucket": "2024-01", "count": 2},
        {"bucket": "2024-02", "count": 1},
    ]
    assert summary["source_project_counts"] == {"max": 2, "me": 1}
    assert summary["content_type_counts"] == {"finding": 2, "insight": 1}
    assert summary["tag_counts"] == {"grid": 2, "solar": 2, "storage": 1}
    assert summary["first_seen_at"] == "2024-01-10T09:00:00+00:00"
    assert summary["last_seen_at"] == "2024-02-01T09:00:00+00:00"


def test_collection_activity_summary_orders_buckets_chronologically_after_limit(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.create_collection("recent")
        for day in (5, 1, 4, 2, 3):
            unit = store.insert_unit(
                _unit(
                    f"day-{day}",
                    f"Day {day}",
                    created_at=f"2024-01-{day:02d}T12:00:00+00:00",
                )
            )
            store.add_unit_to_collection("recent", unit.id)

        summary = store.collection_activity_summary("recent", bucket="day", limit=3)
    finally:
        store.close()

    assert summary["buckets"] == [
        {"bucket": "2024-01-03", "count": 1},
        {"bucket": "2024-01-04", "count": 1},
        {"bucket": "2024-01-05", "count": 1},
    ]


@pytest.mark.parametrize(
    ("bucket", "expected"),
    [
        ("day", [{"bucket": "2024-01-15", "count": 2}]),
        ("week", [{"bucket": "2024-01-15", "count": 2}]),
        ("month", [{"bucket": "2024-01", "count": 2}]),
        ("year", [{"bucket": "2024", "count": 2}]),
    ],
)
def test_collection_activity_summary_supports_bucket_granularities(
    tmp_path,
    bucket,
    expected,
):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.create_collection("granularity")
        for index, timestamp in enumerate(
            ("2024-01-15T01:00:00+00:00", "2024-01-15T23:00:00+00:00")
        ):
            unit = store.insert_unit(
                _unit(f"granularity-{index}", f"Granularity {index}", created_at=timestamp)
            )
            store.add_unit_to_collection("granularity", unit.id)

        summary = store.collection_activity_summary("granularity", bucket=bucket)
    finally:
        store.close()

    assert summary["buckets"] == expected


def test_collection_activity_summary_can_bucket_by_ingested_or_updated_at(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.create_collection("fields")
        first = store.insert_unit(
            _unit("first", "First", created_at="2024-01-01T00:00:00+00:00")
        )
        second = store.insert_unit(
            _unit("second", "Second", created_at="2024-01-02T00:00:00+00:00")
        )
        _set_unit_times(
            store,
            first.id,
            created_at="2024-01-01T00:00:00+00:00",
            ingested_at="2024-02-01T00:00:00+00:00",
            updated_at="2024-03-01T00:00:00+00:00",
        )
        _set_unit_times(
            store,
            second.id,
            created_at="2024-01-02T00:00:00+00:00",
            ingested_at="2024-02-15T00:00:00+00:00",
            updated_at="2024-04-01T00:00:00+00:00",
        )
        store.add_unit_to_collection("fields", first.id)
        store.add_unit_to_collection("fields", second.id)

        ingested = store.collection_activity_summary("fields", field="ingested_at")
        updated = store.collection_activity_summary("fields", field="updated_at")
    finally:
        store.close()

    assert ingested["buckets"] == [{"bucket": "2024-02", "count": 2}]
    assert updated["buckets"] == [
        {"bucket": "2024-03", "count": 1},
        {"bucket": "2024-04", "count": 1},
    ]


def test_collection_activity_summary_rejects_invalid_bucket_field_and_limit(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        with pytest.raises(ValueError, match="Unsupported collection activity bucket"):
            store.collection_activity_summary("anything", bucket="quarter")

        with pytest.raises(ValueError, match="Unsupported collection activity field"):
            store.collection_activity_summary("anything", field="deleted_at")

        with pytest.raises(ValueError, match="limit must be a positive integer"):
            store.collection_activity_summary("anything", limit=0)
    finally:
        store.close()
