"""Tests for duplicate external URL reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    for candidate in (
        Path(path),
        Path(path).with_name(Path(path).name + "-wal"),
        Path(path).with_name(Path(path).name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    source_entity_type: str = "insight",
    title: str | None = None,
    content: str = "",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=title or f"Title {unit_id}",
        content=content or f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def compact(unit_id: str, title: str | None = None) -> dict:
    return {
        "id": unit_id,
        "source_project": "max",
        "source_id": unit_id,
        "source_entity_type": "insight",
        "title": title or f"Title {unit_id}",
    }


def test_find_duplicate_external_urls_groups_metadata_url_fields(store: Store):
    store.insert_unit(
        unit(
            "canonical",
            metadata={"canonical_url": "https://example.com/report"},
        )
    )
    store.insert_unit(
        unit(
            "source",
            metadata={"citation": {"source_url": "https://example.com/report/"}},
        )
    )
    store.insert_unit(unit("unique", metadata={"external_url": "https://example.com/unique"}))

    assert store.find_duplicate_external_urls() == [
        {
            "url": "https://example.com/report",
            "count": 2,
            "units": [compact("canonical"), compact("source")],
        }
    ]


def test_find_duplicate_external_urls_extracts_content_urls(store: Store):
    store.insert_unit(unit("first", content="Read https://example.com/research."))
    store.insert_unit(unit("second", content="Mirror: (https://example.com/research/)"))
    store.insert_unit(unit("other", content="Only https://example.com/other appears once"))

    assert store.find_duplicate_external_urls() == [
        {
            "url": "https://example.com/research",
            "count": 2,
            "units": [compact("first"), compact("second")],
        }
    ]


def test_find_duplicate_external_urls_groups_normalized_variants_deterministically(
    store: Store,
):
    store.insert_unit(
        unit(
            "alpha",
            metadata={"url": "https://Example.com:443/path/#fragment"},
        )
    )
    store.insert_unit(
        unit(
            "beta",
            content="URL: HTTPS://example.com/path",
        )
    )
    store.insert_unit(
        unit(
            "gamma",
            metadata={"link": "https://example.com/path?version=2"},
        )
    )

    assert store.find_duplicate_external_urls() == [
        {
            "url": "https://example.com/path",
            "count": 2,
            "units": [compact("alpha"), compact("beta")],
        }
    ]


def test_find_duplicate_external_urls_deduplicates_repeated_url_within_one_unit(
    store: Store,
):
    store.insert_unit(
        unit(
            "repeated",
            content="https://example.com/dup https://example.com/dup/",
            metadata={
                "url": "https://example.com/dup",
                "links": [{"url": "https://example.com/dup/"}],
            },
        )
    )
    store.insert_unit(unit("other", metadata={"external_url": "https://example.com/dup"}))

    assert store.find_duplicate_external_urls() == [
        {
            "url": "https://example.com/dup",
            "count": 2,
            "units": [compact("other"), compact("repeated")],
        }
    ]


def test_find_duplicate_external_urls_limit_controls_duplicate_groups(store: Store):
    for unit_id in ("a1", "a2", "a3"):
        store.insert_unit(unit(unit_id, metadata={"url": "https://example.com/a"}))
    for unit_id in ("b1", "b2"):
        store.insert_unit(unit(unit_id, metadata={"url": "https://example.com/b"}))
    for unit_id in ("c1", "c2"):
        store.insert_unit(unit(unit_id, metadata={"url": "https://example.com/c"}))

    assert [
        row["url"] for row in store.find_duplicate_external_urls(limit=2)
    ] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert store.find_duplicate_external_urls(limit=0) == []


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_find_duplicate_external_urls_rejects_invalid_limits(store: Store, limit: object):
    with pytest.raises(ValueError, match="non-negative integer"):
        store.find_duplicate_external_urls(limit=limit)  # type: ignore[arg-type]
