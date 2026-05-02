from __future__ import annotations

import pytest

from graph.rag import detect_contradiction_cues
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, title: str, content: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
    )


def test_detect_contradiction_cues_scans_dict_results_case_insensitively():
    results = [
        {
            "id": "unit-a",
            "title": "Replication update",
            "content": "The original finding FAILS TO REPLICATE in the larger cohort.",
        }
    ]

    assert detect_contradiction_cues(results) == [
        {
            "unit_id": "unit-a",
            "title": "Replication update",
            "cue": "fails to replicate",
            "text_key": "content",
            "snippet": "The original finding FAILS TO REPLICATE in the larger cohort.",
            "score": 4.6,
        }
    ]


def test_detect_contradiction_cues_scans_knowledge_unit_objects():
    results = [unit("unit-b", "Retracted trial", "The journal posted a retraction after review.")]

    assert detect_contradiction_cues(results) == [
        {
            "unit_id": "unit-b",
            "title": "Retracted trial",
            "cue": "retracted",
            "text_key": "title",
            "snippet": "Retracted trial",
            "score": 10.0,
        }
    ]


def test_detect_contradiction_cues_uses_configured_text_keys_and_nested_units():
    results = [
        {
            "unit": {
                "id": "unit-c",
                "title": "Nested result",
                "abstract": "A later correction challenges the initial interpretation.",
            }
        }
    ]

    assert detect_contradiction_cues(results, text_keys=("abstract",)) == [
        {
            "unit_id": "unit-c",
            "title": "Nested result",
            "cue": "correction",
            "text_key": "abstract",
            "snippet": "A later correction challenges the initial interpretation.",
            "score": 6.8,
        }
    ]


def test_detect_contradiction_cues_bounds_snippets_and_keeps_matched_cue():
    content = f"{'before ' * 40}however {'after ' * 40}"

    results = detect_contradiction_cues(
        [{"id": "unit-d", "title": "Long text", "content": content}]
    )

    assert len(results[0]["snippet"]) <= 160
    assert "however" in results[0]["snippet"]


def test_detect_contradiction_cues_ranking_is_deterministic_and_limited():
    results = [
        {
            "id": "unit-c",
            "title": "Commentary",
            "content": "However, the result is useful but preliminary.",
        },
        {
            "id": "unit-a",
            "title": "Retraction notice",
            "content": "The paper was retracted after a correction.",
        },
        {
            "id": "unit-b",
            "title": "Challenge",
            "content": "This challenges the original result.",
        },
    ]

    first = detect_contradiction_cues(results, limit=2)
    second = detect_contradiction_cues(reversed(results), limit=2)

    assert first == second
    assert [item["unit_id"] for item in first] == ["unit-a", "unit-b"]
    assert [item["score"] for item in first] == [13.6, 6.4]


def test_detect_contradiction_cues_returns_empty_list_when_no_cues_match():
    assert (
        detect_contradiction_cues(
            [{"id": "unit-a", "title": "Calm result", "content": "No review markers here."}]
        )
        == []
    )


@pytest.mark.parametrize("limit", [0, -1, "2", True])
def test_detect_contradiction_cues_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        detect_contradiction_cues([], limit=limit)
