from __future__ import annotations

from datetime import datetime, timezone

from graph.store.db import Store
from graph.store.unit_source_entity_type_mix_summary import summarize_unit_source_entity_type_mix
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str,
    source_entity_type: str,
    content_type: ContentType,
    created_at: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=unit_id,
        content=f"Content {unit_id}",
        content_type=content_type,
        created_at=dt(created_at),
    )


def test_summarize_unit_source_entity_type_mix_counts_sorted_groups(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(
            unit(
                "max-page-2",
                source_project=SourceProject.MAX,
                source_entity_type="page",
                content_type=ContentType.FINDING,
                created_at="2026-01-02T00:00:00",
            )
        )
        store.insert_unit(
            unit(
                "bookmarks-page-1",
                source_project=SourceProject.BOOKMARKS,
                source_entity_type="page",
                content_type=ContentType.INSIGHT,
                created_at="2026-01-03T00:00:00",
            )
        )
        store.insert_unit(
            unit(
                "max-page-1",
                source_project=SourceProject.MAX,
                source_entity_type="page",
                content_type=ContentType.INSIGHT,
                created_at="2026-01-01T00:00:00",
            )
        )
        store.insert_unit(
            unit(
                "max-comment-1",
                source_project=SourceProject.MAX,
                source_entity_type="comment",
                content_type=ContentType.ARTIFACT,
                created_at="2026-01-04T00:00:00",
            )
        )

        summary = summarize_unit_source_entity_type_mix(store)
    finally:
        store.close()

    assert [(row["source_project"], row["source_entity_type"]) for row in summary["rows"]] == [
        ("bookmarks", "page"),
        ("max", "comment"),
        ("max", "page"),
    ]
    assert summary["rows"][2] == {
        "source_project": "max",
        "source_entity_type": "page",
        "unit_count": 2,
        "example_unit_ids": ["max-page-1", "max-page-2"],
        "content_type_counts": {"finding": 1, "insight": 1},
        "earliest_created_at": "2026-01-01T00:00:00+00:00",
        "latest_created_at": "2026-01-02T00:00:00+00:00",
    }


def test_summarize_unit_source_entity_type_mix_bounds_examples_and_skips_bad_timestamps(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(
            unit(
                "one",
                source_project=SourceProject.CSV,
                source_entity_type="row",
                content_type=ContentType.METADATA,
                created_at="2026-01-01T00:00:00",
            )
        )
        store.insert_unit(
            unit(
                "two",
                source_project=SourceProject.CSV,
                source_entity_type="row",
                content_type=ContentType.METADATA,
                created_at="2026-01-02T00:00:00",
            )
        )
        store.conn.execute("UPDATE knowledge_units SET created_at = ? WHERE id = ?", ("not-a-date", "two"))
        store.conn.commit()

        summary = summarize_unit_source_entity_type_mix(store, example_limit=1)
    finally:
        store.close()

    assert summary["rows"] == [
        {
            "source_project": "csv",
            "source_entity_type": "row",
            "unit_count": 2,
            "example_unit_ids": ["one"],
            "content_type_counts": {"metadata": 2},
            "earliest_created_at": "2026-01-01T00:00:00+00:00",
            "latest_created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
