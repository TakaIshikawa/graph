from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    metadata: dict | None = None,
    content: str = "",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Unit {unit_id}",
        content=content or f"Content {unit_id}",
        metadata=metadata or {},
    )


def test_external_domain_summary_empty_store_returns_empty(store: Store):
    assert store.external_domain_summary() == []


def test_external_domain_summary_normalizes_metadata_and_content_domains(store: Store):
    store.insert_unit(
        _unit(
            "a",
            metadata={"source_url": "HTTPS://Example.COM/path?x=1"},
            content="See https://Docs.Example.com/guide.",
        )
    )
    store.insert_unit(
        _unit(
            "b",
            source_project=SourceProject.PRESENCE,
            metadata={"link": "example.com/other"},
            content="Repeated https://example.com/path",
        )
    )

    assert store.external_domain_summary() == [
        {
            "domain": "example.com",
            "unit_count": 2,
            "source_project_counts": {"max": 1, "presence": 1},
            "metadata_key_counts": {"content": 1, "link": 1, "source_url": 1},
            "example_unit_ids": ["a", "b"],
        },
        {
            "domain": "docs.example.com",
            "unit_count": 1,
            "source_project_counts": {"max": 1},
            "metadata_key_counts": {"content": 1},
            "example_unit_ids": ["a"],
        },
    ]


def test_external_domain_summary_restricts_metadata_keys_and_content_scanning(store: Store):
    store.insert_unit(
        _unit(
            "a",
            metadata={
                "source": {"url": "https://alpha.test/a"},
                "canonical_url": "https://beta.test/b",
            },
            content="https://content.test/c",
        )
    )
    store.insert_unit(_unit("b", metadata={"source": {"url": "https://alpha.test/b"}}))

    assert store.external_domain_summary(
        metadata_keys=["source.url"],
        include_content_urls=False,
    ) == [
        {
            "domain": "alpha.test",
            "unit_count": 2,
            "source_project_counts": {"max": 2},
            "metadata_key_counts": {"source.url": 2},
            "example_unit_ids": ["a", "b"],
        },
    ]


def test_external_domain_summary_filters_and_limits(store: Store):
    store.insert_unit(_unit("a", metadata={"url": "https://alpha.test/a"}))
    store.insert_unit(_unit("b", metadata={"url": "https://alpha.test/b"}))
    store.insert_unit(_unit("c", metadata={"url": "https://beta.test/c"}))

    assert [row["domain"] for row in store.external_domain_summary(min_unit_count=2)] == [
        "alpha.test"
    ]
    assert [row["domain"] for row in store.external_domain_summary(limit=1)] == ["alpha.test"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metadata_keys": "url"}, "metadata_keys must be a sequence"),
        ({"metadata_keys": [""]}, "metadata_keys must be a sequence"),
        ({"include_content_urls": "yes"}, "include_content_urls must be a boolean"),
        ({"limit": 0}, "limit must be a positive integer or None"),
        ({"min_unit_count": 0}, "min_unit_count must be a positive integer"),
    ],
)
def test_external_domain_summary_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.external_domain_summary(**kwargs)
