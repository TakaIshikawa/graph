from __future__ import annotations

from graph.rag.result_deduplication_plan import build_result_deduplication_plan


def test_result_deduplication_plan_groups_normalized_urls():
    plan = build_result_deduplication_plan(
        [
            {"id": "b", "url": "HTTPS://Example.com/Doc/"},
            {"id": "a", "url": "https://example.com/Doc"},
            {"id": "c", "url": "https://other.test/doc"},
        ]
    )

    assert plan["canonical_result_ids"] == ["a"]
    assert plan["dropped_result_ids"] == ["b"]
    assert plan["duplicate_count"] == 1
    assert plan["retention_reasons"] == {"a": "matching_url_or_path"}


def test_result_deduplication_plan_groups_title_content_without_urls():
    plan = build_result_deduplication_plan(
        [
            {"id": "r1", "title": "Market Update", "content": "Rates rose today."},
            {"id": "r2", "title": "market update", "text": "Rates rose today!"},
        ]
    )

    assert plan["duplicate_groups"][0]["canonical_result_id"] == "r1"
    assert plan["duplicate_groups"][0]["duplicate_result_ids"] == ["r2"]
