from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "graph.db"
    s = Store(str(path))
    yield s
    s.close()
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def unit(
    unit_id: str,
    title: str,
    *,
    tags: list[str],
    source_project: SourceProject | str = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_preview_tag_rename_reports_units_without_mutating(store: Store):
    store.insert_unit(unit("gamma", "Gamma", tags=["topic", "review"]))
    store.insert_unit(unit("alpha", "Alpha", tags=["topic", "draft"], source_project="custom"))

    preview = store.preview_tag_rename("topic", "theme")

    assert preview == {
        "old_tag": "topic",
        "new_tag": "theme",
        "affected_count": 2,
        "returned_units": [
            {
                "id": "alpha",
                "title": "Alpha",
                "source_project": "custom",
                "source_id": "source-alpha",
                "source_entity_type": "note",
                "content_type": "insight",
                "before_tags": ["topic", "draft"],
                "after_tags": ["theme", "draft"],
            },
            {
                "id": "gamma",
                "title": "Gamma",
                "source_project": "max",
                "source_id": "source-gamma",
                "source_entity_type": "note",
                "content_type": "insight",
                "before_tags": ["topic", "review"],
                "after_tags": ["theme", "review"],
            },
        ],
    }
    assert store.get_unit("alpha").tags == ["topic", "draft"]  # type: ignore[union-attr]
    assert store.get_unit("gamma").tags == ["topic", "review"]  # type: ignore[union-attr]


def test_preview_tag_rename_dedupes_when_new_tag_already_exists(store: Store):
    store.insert_unit(unit("first", "First", tags=["old", "new", "keep"]))
    store.insert_unit(unit("second", "Second", tags=["new", "old", "keep"]))

    preview = store.preview_tag_rename("old", "new")

    assert [unit["after_tags"] for unit in preview["returned_units"]] == [
        ["new", "keep"],
        ["new", "keep"],
    ]
    assert store.get_unit("first").tags == ["old", "new", "keep"]  # type: ignore[union-attr]
    assert store.get_unit("second").tags == ["new", "old", "keep"]  # type: ignore[union-attr]


def test_preview_tag_rename_is_exact_match_and_missing_old_tag_is_empty(store: Store):
    store.insert_unit(unit("alpha", "Alpha", tags=["Topic", "topic-extra"]))

    preview = store.preview_tag_rename("topic", "theme")

    assert preview == {
        "old_tag": "topic",
        "new_tag": "theme",
        "affected_count": 0,
        "returned_units": [],
    }


def test_preview_tag_rename_limit_keeps_total_count_and_limits_returned_units(store: Store):
    store.insert_unit(unit("gamma", "Gamma", tags=["old"]))
    store.insert_unit(unit("alpha", "Alpha", tags=["old"]))
    store.insert_unit(unit("beta", "Beta", tags=["old"]))

    preview = store.preview_tag_rename("old", "new", limit=2)
    empty_limit = store.preview_tag_rename("old", "new", limit=0)

    assert preview["affected_count"] == 3
    assert [unit["id"] for unit in preview["returned_units"]] == ["alpha", "beta"]
    assert empty_limit == {
        "old_tag": "old",
        "new_tag": "new",
        "affected_count": 3,
        "returned_units": [],
    }


def test_preview_tag_rename_noop_same_tag_reports_no_affected_units(store: Store):
    store.insert_unit(unit("alpha", "Alpha", tags=["topic", "draft"]))

    preview = store.preview_tag_rename("topic", "topic")

    assert preview == {
        "old_tag": "topic",
        "new_tag": "topic",
        "affected_count": 0,
        "returned_units": [],
    }


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_preview_tag_rename_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        store.preview_tag_rename("old", "new", limit=limit)
