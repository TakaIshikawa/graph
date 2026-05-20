from __future__ import annotations

from types import SimpleNamespace

from graph.rag.context_conflict_flags import flag_context_conflicts
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, content: str) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="notes",
        source_id=f"source-{unit_id}",
        source_entity_type="claim",
        title=f"Title {unit_id}",
        content=content,
        metadata={},
        tags=[],
    )


def test_context_conflict_flags_opposing_cues_across_shared_query_terms():
    payload = flag_context_conflicts(
        "battery storage cost",
        [
            {"id": "a", "content": "Battery storage cost increased in the latest filing."},
            {"id": "b", "content": "Battery storage cost decreased after procurement changes."},
        ],
    )

    assert payload["query_terms"] == ["battery", "cost", "storage"]
    assert payload["conflict_count"] >= 1
    row = next(flag for flag in payload["flags"] if flag["term"] == "cost")
    assert row["conflict_type"] == "opposing_cues"
    assert row["cue_pair"] == "decrease / increase"
    assert row["result_ids"] == ["a", "b"]
    assert row["confidence"] == "high"


def test_context_conflict_flags_accepts_objects_tuples_and_nested_units():
    wrapped = SimpleNamespace(id="wrapper", unit=unit("nested", "The API is supported for export jobs."))
    payload = flag_context_conflicts(
        "api export",
        [
            wrapped,
            ({"unit_id": "tuple", "snippet": "The API is unsupported for export jobs."}, 0.8),
        ],
    )

    assert payload["flags"] == [
        {
            "term": "api",
            "conflict_type": "opposing_cues",
            "cue_pair": "supported / unsupported",
            "result_ids": ["tuple", "wrapper"],
            "confidence": "high",
            "evidence": [
                {"result_id": "tuple", "snippet": "The API is unsupported for export jobs."},
                {"result_id": "wrapper", "snippet": "The API is supported for export jobs. Title nested"},
            ],
        },
        {
            "term": "export",
            "conflict_type": "opposing_cues",
            "cue_pair": "supported / unsupported",
            "result_ids": ["tuple", "wrapper"],
            "confidence": "high",
            "evidence": [
                {"result_id": "tuple", "snippet": "The API is unsupported for export jobs."},
                {"result_id": "wrapper", "snippet": "The API is supported for export jobs. Title nested"},
            ],
        },
    ]


def test_context_conflict_flags_numeric_disagreement_near_matching_terms():
    payload = flag_context_conflicts(
        "retention rate",
        [
            {"id": "a", "content": "Retention rate was 72% in the cohort."},
            {"id": "b", "content": "Retention rate was 65% in the cohort."},
        ],
    )

    row = next(flag for flag in payload["flags"] if flag["conflict_type"] == "numeric_disagreement")
    assert row["term"] == "rate"
    assert row["result_ids"] == ["a", "b"]
    assert row["values"] == ["65%", "72%"]
    assert row["confidence"] == "medium"


def test_context_conflict_flags_ignores_unrelated_opposing_cues_without_shared_query_terms():
    payload = flag_context_conflicts(
        "battery storage",
        [
            {"id": "a", "content": "Battery storage increased."},
            {"id": "b", "content": "Hiring decreased."},
        ],
    )

    assert payload["flags"] == []
