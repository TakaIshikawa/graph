from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.store.source_crawl_depth_summary import summarize_source_crawl_depth
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def unit(unit_id: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="page",
        title=unit_id,
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def by_id(summary: dict) -> dict:
    return {row["unit_id"]: row for row in summary["sources"]}


def test_summarize_source_crawl_depth_derives_depths_and_host_counts(store: Store):
    store.insert_unit(unit("root", {"url": "https://example.com/"}))
    store.insert_unit(
        unit("child", {"url": "https://example.com/a", "parent_url": "https://example.com/"})
    )
    store.insert_unit(
        unit("grandchild", {"url": "https://example.com/b", "referrer_url": "https://example.com/a"})
    )
    store.insert_unit(unit("other-root", {"url": "https://other.test/start"}))

    summary = summarize_source_crawl_depth(store)
    rows = by_id(summary)

    assert rows["root"]["depth"] == 0
    assert rows["child"]["depth"] == 1
    assert rows["grandchild"]["depth"] == 2
    assert rows["grandchild"]["status"] == "resolved"
    assert summary["host_depth_counts"] == [
        {"host": "example.com", "depth": "0", "count": 1},
        {"host": "example.com", "depth": "1", "count": 1},
        {"host": "example.com", "depth": "2", "count": 1},
        {"host": "other.test", "depth": "0", "count": 1},
    ]


def test_summarize_source_crawl_depth_marks_unresolved_parent_and_cycles(store: Store):
    store.insert_unit(
        unit("orphan", {"url": "https://example.com/orphan", "parent_url": "https://example.com/missing"})
    )
    store.insert_unit(unit("cycle-a", {"url": "https://example.com/a", "parent_url": "https://example.com/b"}))
    store.insert_unit(unit("cycle-b", {"url": "https://example.com/b", "parent_url": "https://example.com/a"}))

    rows = by_id(summarize_source_crawl_depth(store))

    assert rows["orphan"]["depth"] is None
    assert rows["orphan"]["status"] == "unresolved_parent"
    assert rows["cycle-a"]["depth"] is None
    assert rows["cycle-a"]["status"] == "cycle"
    assert rows["cycle-b"]["status"] == "cycle"


def test_summarize_source_crawl_depth_normalizes_urls_and_supports_custom_keys(store: Store):
    store.insert_unit(unit("root", {"source": {"url": "HTTPS://Example.COM:443/root#section"}}))
    store.insert_unit(
        unit(
            "child",
            {
                "source": {"url": "https://example.com/root/child"},
                "source_parent": {"url": "https://example.com/root"},
            },
        )
    )

    rows = by_id(
        summarize_source_crawl_depth(
            store,
            url_keys=["source.url"],
            parent_url_keys=["source_parent.url"],
        )
    )

    assert rows["root"]["url"] == "https://example.com/root"
    assert rows["child"]["depth"] == 1


def test_summarize_source_crawl_depth_validates_key_sequences(store: Store):
    with pytest.raises(ValueError, match="url_keys"):
        summarize_source_crawl_depth(store, url_keys="url")  # type: ignore[arg-type]
