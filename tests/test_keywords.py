from __future__ import annotations

from graph.rag import extract_keywords
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, title: str, content: str, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def by_keyword(results: list[dict], keyword: str) -> dict:
    return next(item for item in results if item["keyword"] == keyword)


def test_extract_keywords_boosts_title_matches_over_content_matches():
    units = [
        unit("unit-a", "Solar storage roadmap", "storage economics improve"),
        unit("unit-b", "Market update", "battery battery battery"),
    ]

    results = extract_keywords(units)

    assert results[0]["keyword"] == "storage"
    assert by_keyword(results, "storage") == {
        "keyword": "storage",
        "score": 4,
        "count": 2,
        "sources": {"title": 1, "content": 1, "tags": 0},
        "unit_ids": ["unit-a"],
    }


def test_extract_keywords_can_include_or_ignore_tags():
    units = [
        unit("unit-a", "Storage note", "plain content", ["Solar", "Grid-Scale"]),
        unit("unit-b", "Finance note", "plain content", ["solar"]),
    ]

    with_tags = extract_keywords(units)
    without_tags = extract_keywords(units, include_tags=False)

    assert by_keyword(with_tags, "solar") == {
        "keyword": "solar",
        "score": 4,
        "count": 2,
        "sources": {"title": 0, "content": 0, "tags": 2},
        "unit_ids": ["unit-a", "unit-b"],
    }
    assert "solar" not in {item["keyword"] for item in without_tags}
    assert "grid" in {item["keyword"] for item in with_tags}
    assert "scale" in {item["keyword"] for item in with_tags}


def test_extract_keywords_excludes_common_and_custom_stopwords():
    units = [
        unit(
            "unit-a",
            "The Solar Plan",
            "The plan depends on market demand and solar demand.",
            ["market"],
        )
    ]

    results = extract_keywords(units, stopwords={"solar", "market"})
    keywords = {item["keyword"] for item in results}

    assert "the" not in keywords
    assert "and" not in keywords
    assert "solar" not in keywords
    assert "market" not in keywords
    assert by_keyword(results, "demand")["count"] == 2


def test_extract_keywords_honors_min_count_and_limit():
    units = [
        unit("unit-a", "Alpha Beta", "beta gamma"),
        unit("unit-b", "Delta", "gamma epsilon"),
    ]

    results = extract_keywords(units, min_count=2, limit=1)

    assert results == [
        {
            "keyword": "beta",
            "score": 4,
            "count": 2,
            "sources": {"title": 1, "content": 1, "tags": 0},
            "unit_ids": ["unit-a"],
        }
    ]


def test_extract_keywords_orders_equal_scores_deterministically():
    units = [
        unit("unit-b", "Beta", "alpha"),
        unit("unit-a", "Alpha", "beta"),
    ]

    first = extract_keywords(units)
    second = extract_keywords(reversed(units))

    assert [item["keyword"] for item in first[:2]] == ["alpha", "beta"]
    assert [item["keyword"] for item in second[:2]] == ["alpha", "beta"]
    assert by_keyword(first, "alpha")["unit_ids"] == ["unit-a", "unit-b"]
