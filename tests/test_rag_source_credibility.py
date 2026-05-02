from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from graph.rag import score_source_credibility


NOW = datetime(2026, 5, 3, tzinfo=timezone.utc)


def by_id(scored: list[dict], result_id: str) -> dict:
    return next(item for item in scored if item["id"] == result_id)


def test_score_source_credibility_returns_one_entry_per_input_without_mutation():
    results = [
        {
            "id": "trusted",
            "title": "Research note",
            "url": "https://example.org/report",
            "published_at": "2026-04-20",
            "confidence": 0.8,
            "utility_score": 0.7,
            "citation_count": 4,
            "content": " ".join(["evidence"] * 90),
        },
        {
            "id": "thin",
            "title": "Short",
            "metadata": {"source_url": "https://blog.test/post"},
            "created_at": "2026-04-19",
            "content": "tiny",
        },
    ]
    original = deepcopy(results)

    scored = score_source_credibility(
        results,
        trusted_domains=["example.org"],
        now=NOW,
    )

    assert results == original
    assert {item["id"] for item in scored} == {"trusted", "thin"}
    assert all(0.0 <= item["score"] <= 1.0 for item in scored)
    assert all(item["band"] in {"low", "medium", "high"} for item in scored)
    assert all(isinstance(item["reasons"], list) and item["reasons"] for item in scored)


def test_trusted_domain_boosts_matching_domains_and_subdomains():
    scored = score_source_credibility(
        [
            {
                "id": "trusted",
                "title": "Trusted",
                "source_url": "https://docs.example.com/a",
                "published_at": "2026-04-20",
                "content": " ".join(["useful"] * 30),
            },
            {
                "id": "ordinary",
                "title": "Ordinary",
                "source_url": "https://ordinary.test/a",
                "published_at": "2026-04-20",
                "content": " ".join(["useful"] * 30),
            },
        ],
        trusted_domains=["example.com"],
        now=NOW,
    )

    trusted = by_id(scored, "trusted")
    ordinary = by_id(scored, "ordinary")
    assert trusted["domain"] == "docs.example.com"
    assert trusted["score"] > ordinary["score"]
    assert "trusted domain: docs.example.com" in trusted["reasons"]


def test_recent_timestamps_score_higher_than_stale_items():
    scored = score_source_credibility(
        [
            {
                "id": "recent",
                "title": "Recent",
                "domain": "source.test",
                "published_at": "2026-05-01",
                "content": " ".join(["context"] * 30),
            },
            {
                "id": "stale",
                "title": "Stale",
                "domain": "source.test",
                "published_at": "2024-05-01",
                "content": " ".join(["context"] * 30),
            },
        ],
        recency_half_life_days=90,
        now=NOW,
    )

    recent = by_id(scored, "recent")
    stale = by_id(scored, "stale")
    assert recent["score"] > stale["score"]
    assert "recent timestamp" in recent["reasons"]
    assert "stale timestamp" in stale["reasons"]


def test_missing_metadata_is_handled_with_fallbacks_and_reasons():
    scored = score_source_credibility([{"content": "brief"}], now=NOW)

    assert scored == [
        {
            "id": "result-1",
            "title": None,
            "domain": None,
            "score": pytest.approx(0.27),
            "band": "low",
            "reasons": [
                "baseline source credibility",
                "missing source domain",
                "missing usable timestamp",
                "missing title",
                "thin content",
            ],
        }
    ]


def test_scoring_sort_is_deterministic_across_input_order():
    results = [
        {
            "id": "b",
            "title": "Beta",
            "domain": "same.test",
            "published_at": "2026-01-01",
            "content": " ".join(["same"] * 20),
        },
        {
            "id": "a",
            "title": "Alpha",
            "domain": "same.test",
            "published_at": "2026-01-01",
            "content": " ".join(["same"] * 20),
        },
    ]

    first = score_source_credibility(results, now=NOW)
    second = score_source_credibility(reversed(results), now=NOW)

    assert first == second
    assert [item["id"] for item in first] == ["a", "b"]


def test_citation_confidence_utility_and_content_signals_generate_reasons():
    scored = score_source_credibility(
        [
            {
                "id": "paper",
                "title": "Paper",
                "metadata": {
                    "source_url": "https://journal.example/paper",
                    "published_at": "2026-04-01",
                    "doi": "10.1000/example",
                },
                "confidence": "85%",
                "utility_score": 0.9,
                "content": " ".join(["method"] * 85),
            }
        ],
        now=NOW,
    )

    reasons = scored[0]["reasons"]
    assert "confidence 0.85" in reasons
    assert "utility score 0.90" in reasons
    assert "stable citation identifier present" in reasons
    assert "substantive content" in reasons


def test_score_source_credibility_is_importable_from_graph_rag():
    assert callable(score_source_credibility)


@pytest.mark.parametrize("half_life", [0, -1, "180", True])
def test_score_source_credibility_validates_half_life(half_life):
    with pytest.raises(ValueError, match="recency_half_life_days must be a positive number"):
        score_source_credibility([], recency_half_life_days=half_life)
