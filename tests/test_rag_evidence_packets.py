from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.rag.evidence_packets import build_evidence_packets
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

BASE_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, **kwargs) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=kwargs.get("source_project", SourceProject.MAX),
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=kwargs.get("title", f"Title {unit_id}"),
        content=kwargs.get("content", f"Content about retrieval {unit_id}"),
        content_type=ContentType.INSIGHT,
        tags=kwargs.get("tags", []),
        metadata=kwargs.get("metadata", {}),
        created_at=kwargs.get("created_at", BASE_TIME),
        ingested_at=kwargs.get("created_at", BASE_TIME),
        updated_at=kwargs.get("created_at", BASE_TIME),
    )


def test_build_evidence_packets_normalizes_units_and_mappings():
    packets = build_evidence_packets(
        [
            (
                unit(
                    "unit-a",
                    title="Alpha",
                    tags=["rag", "evidence"],
                    metadata={"url": "https://example.com/a", "published_at": "2026-05-03"},
                ),
                0.8,
            ),
            {
                "id": "map-b",
                "title": "Beta",
                "content": "Beta result explains retrieval quality.",
                "source_project": "pinboard",
                "source_id": "pin-b",
                "source_entity_type": "bookmark",
                "tags": ["bookmark"],
                "doi": "10.1000/example",
                "created_at": "2026-05-02T09:00:00Z",
                "score": 0.4,
            },
        ],
        query="retrieval",
    )

    assert [packet["id"] for packet in packets] == ["unit-a", "map-b"]
    assert packets[0]["source_project"] == "max"
    assert packets[0]["citation_fields"]["url"] == "https://example.com/a"
    assert packets[0]["domain"] == "example.com"
    assert packets[0]["date_fields"]["published_at"] == "2026-05-03T00:00:00+00:00"
    assert packets[0]["tags"] == ["evidence", "rag"]
    assert packets[0]["missing_citation_warnings"] == []
    assert packets[0]["evidence_strength"] > packets[1]["evidence_strength"]


def test_build_evidence_packets_warns_about_missing_citations_and_sorts_stably():
    packets = build_evidence_packets(
        [
            {"id": "b", "title": "Beta", "content": "Body", "score": 0.5},
            {"id": "a", "title": "Alpha", "content": "Body", "score": 0.5},
        ]
    )

    assert [packet["id"] for packet in packets] == ["a", "b"]
    assert packets[0]["missing_citation_warnings"] == ["missing_citation"]
    assert packets[0]["citation_fields"] == {}


def test_build_evidence_packets_handles_missing_metadata_and_limit_zero():
    assert build_evidence_packets([{"id": "a", "title": "A"}], limit=0) == []

    packet = build_evidence_packets([{"id": "a", "title": "A"}])[0]

    assert packet["snippet"] is None
    assert packet["date_fields"] == {}
    assert packet["source"] is None


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_build_evidence_packets_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        build_evidence_packets([], limit=limit)
