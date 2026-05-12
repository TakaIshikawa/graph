from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.rag.evidence_pack import build_evidence_pack
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, *, metadata=None, confidence=None, utility_score=None):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content about retrieval evidence.",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        confidence=confidence,
        utility_score=utility_score,
        created_at=TIME,
        ingested_at=TIME,
        updated_at=TIME,
    )


def test_evidence_pack_accepts_mixed_mapping_and_unit_inputs():
    pack = build_evidence_pack(
        [
            (
                unit(
                    "a",
                    "Alpha",
                    metadata={"url": "https://example.com/a"},
                    confidence=0.7,
                    utility_score=0.8,
                ),
                0.9,
            ),
            {
                "id": "b",
                "title": "Beta",
                "content": "Beta explains a second source.",
                "source_project": "pinboard",
                "source_id": "pin-b",
                "source_entity_type": "bookmark",
                "url": "https://example.org/b",
                "score": 0.3,
            },
        ],
        limit=2,
        snippet_chars=24,
        summary_char_budget=500,
    )

    assert pack["total_count"] == 2
    assert pack["selected_count"] == 2
    assert pack["source_project_counts"] == {"max": 1, "pinboard": 1}
    assert pack["source_diversity_count"] == 2
    assert [item["id"] for item in pack["evidence"]] == ["a", "b"]
    assert pack["evidence"][0]["snippet"] == "Alpha content about retr"
    assert pack["evidence"][0]["confidence"] == 0.7
    assert pack["evidence"][0]["utility_score"] == 0.8
    assert len(pack["citations"]) == 2
    assert "Alpha" in pack["summary"]


def test_evidence_pack_handles_missing_metadata_and_stable_sorting():
    pack = build_evidence_pack(
        [
            {"id": "b", "title": "Beta"},
            {"id": "a", "title": "Alpha"},
        ],
        limit=2,
    )

    assert [item["id"] for item in pack["evidence"]] == ["b", "a"]
    assert pack["source_project_counts"] == {"unknown": 2}
    assert pack["evidence"][0]["snippet"] is None
    assert pack["evidence"][0]["citation"]


def test_evidence_pack_applies_limit_and_summary_budget():
    pack = build_evidence_pack(
        [
            {"id": "a", "title": "Alpha", "content": "A" * 100, "source_project": "max"},
            {"id": "b", "title": "Beta", "content": "B" * 100, "source_project": "max"},
        ],
        limit=1,
        snippet_chars=80,
        summary_char_budget=30,
    )

    assert pack["selected_count"] == 1
    assert [item["id"] for item in pack["evidence"]] == ["a"]
    assert len(pack["summary"]) <= 30
    assert pack["truncated"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": -1}, "limit"),
        ({"snippet_chars": -1}, "snippet_chars"),
        ({"summary_char_budget": -1}, "summary_char_budget"),
        ({"limit": True}, "limit"),
    ],
)
def test_evidence_pack_validates_budget_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_evidence_pack([], **kwargs)
