from __future__ import annotations

from graph.rag.query_deduplication_requirement import detect_query_deduplication_requirement


def test_detects_duplicate_suppression_key_hash_token_and_merge_requirements():
    result = detect_query_deduplication_requirement(
        "Need duplicate suppression for ingestion, dedupe keys, content hashing, "
        "idempotency tokens for dedupe, and merge duplicate records."
    )

    assert result["has_deduplication_requirement"] is True
    assert result["categories"] == [
        "duplicate_suppression",
        "dedupe_key",
        "content_hash",
        "idempotency_token",
        "duplicate_merge",
    ]
    assert [row["category"] for row in result["requirements"]] == result["categories"]
    assert result["matched_phrases"] == [
        "duplicate suppression",
        "dedupe keys",
        "content hashing",
        "idempotency tokens",
        "merge duplicate records",
    ]
    assert result["confidence"] == "high"


def test_detects_duplicate_merge_and_payload_hash_phrasing():
    result = detect_query_deduplication_requirement(
        "How should the indexer coalesce duplicates using a payload hash?"
    )

    assert result["has_deduplication_requirement"] is True
    assert result["categories"] == ["duplicate_merge", "content_hash"]
    assert result["confidence"] == "high"


def test_exactly_once_delivery_without_dedupe_terms_does_not_match():
    result = detect_query_deduplication_requirement(
        "Explain exactly-once delivery guarantees for Kafka consumers and transactional offsets."
    )

    assert result == {
        "has_deduplication_requirement": False,
        "requirements": [],
        "categories": [],
        "matched_phrases": [],
        "confidence": "none",
    }


def test_idempotency_token_requires_dedupe_context():
    result = detect_query_deduplication_requirement("Should API writes use idempotency tokens for retry safety?")

    assert result["has_deduplication_requirement"] is False
    assert result["categories"] == []


def test_idempotency_token_matches_when_used_for_dedupe():
    result = detect_query_deduplication_requirement("Use idempotency keys to dedupe duplicate record imports.")

    assert result["has_deduplication_requirement"] is True
    assert result["categories"] == ["idempotency_token", "duplicate_suppression"]

