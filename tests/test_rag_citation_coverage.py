from __future__ import annotations

from graph.rag import analyze_citation_coverage
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content="Brief note.",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def test_analyze_citation_coverage_counts_dict_result_evidence():
    report = analyze_citation_coverage(
        [
            {
                "id": "url-result",
                "title": "URL evidence",
                "source": "example",
                "url": "https://example.com/report",
            },
            {
                "id": "doi-result",
                "title": "DOI evidence",
                "metadata": {"doi": "10.1000/example"},
            },
            {
                "id": "citation-result",
                "title": "Citation evidence",
                "metadata": {"citations": ["Smith 2024"]},
            },
            {
                "id": "missing-result",
                "title": "No evidence",
                "source": "local notes",
            },
        ]
    )

    assert report["total_results"] == 4
    assert report["with_citation_count"] == 3
    assert report["with_url_count"] == 1
    assert report["with_identifier_count"] == 1
    assert report["with_explicit_citation_count"] == 1
    assert report["missing_citation_count"] == 1
    assert report["citation_coverage_ratio"] == 0.75
    assert report["missing_citation_ratio"] == 0.25
    assert report["missing_citations"] == [
        {
            "index": 3,
            "id": "missing-result",
            "title": "No evidence",
            "source": "local notes",
        }
    ]

    assert report["results"][0]["has_url"] is True
    assert report["results"][0]["has_citation"] is True
    assert report["results"][1]["identifier_keys"] == ["doi"]
    assert report["results"][2]["citation_keys"] == ["citations"]
    assert report["results"][3]["has_citation"] is False


def test_analyze_citation_coverage_supports_knowledge_unit_results():
    report = analyze_citation_coverage(
        [
            {
                "unit": unit(
                    "unit-a",
                    "ArXiv-backed unit",
                    {"arxiv_id": "2401.12345"},
                )
            },
            {
                "unit": unit(
                    "unit-b",
                    "URL-backed unit",
                    {"source_url": "https://example.com/unit-b"},
                )
            },
            {"unit": unit("unit-c", "Missing unit")},
        ]
    )

    assert report["with_identifier_count"] == 1
    assert report["with_url_count"] == 1
    assert report["with_citation_count"] == 2
    assert report["results"][0]["id"] == "unit-a"
    assert report["results"][0]["source"] == SourceProject.MAX.value
    assert report["results"][0]["identifier_keys"] == ["arxiv_id"]
    assert report["missing_citations"] == [
        {
            "index": 2,
            "id": "unit-c",
            "title": "Missing unit",
            "source": SourceProject.MAX.value,
        }
    ]


def test_analyze_citation_coverage_extends_default_metadata_keys():
    report = analyze_citation_coverage(
        [
            {
                "id": "custom-url",
                "title": "Custom URL key",
                "metadata": {"landing_page": "https://example.com/custom"},
            },
            {
                "id": "custom-citation",
                "title": "Custom citation key",
                "metadata": {"bibliography_entry": "Doe, Example, 2025"},
            },
            {
                "id": "default-still-works",
                "title": "Default DOI key",
                "metadata": {"doi": "10.2000/default"},
            },
        ],
        citation_keys=["bibliography_entry"],
        url_keys=["landing_page"],
    )

    assert report["with_url_count"] == 1
    assert report["with_explicit_citation_count"] == 1
    assert report["with_identifier_count"] == 1
    assert report["with_citation_count"] == 3
    assert report["results"][0]["url_keys"] == ["landing_page"]
    assert report["results"][1]["citation_keys"] == ["bibliography_entry"]
    assert report["results"][2]["identifier_keys"] == ["doi"]


def test_analyze_citation_coverage_accepts_single_custom_key_strings():
    report = analyze_citation_coverage(
        [
            {
                "id": "custom-url",
                "title": "Custom URL key",
                "metadata": {"landing_page": "https://example.com/custom"},
            },
            {
                "id": "custom-citation",
                "title": "Custom citation key",
                "metadata": {"bibliography_entry": "Doe, Example, 2025"},
            },
        ],
        citation_keys="bibliography_entry",
        url_keys="landing_page",
    )

    assert report["with_citation_count"] == 2
    assert report["results"][0]["url_keys"] == ["landing_page"]
    assert report["results"][1]["citation_keys"] == ["bibliography_entry"]


def test_missing_citation_examples_are_sorted_stably_by_context():
    report = analyze_citation_coverage(
        [
            {"id": "zeta", "title": "Zeta", "source": "beta"},
            {"id": "cited", "title": "Cited", "url": "https://example.com"},
            {"id": "alpha-2", "title": "Alpha", "source": "alpha"},
            {"id": "alpha-1", "title": "Alpha", "source": "alpha"},
            {"id": "beta", "title": "Beta", "source": "alpha"},
        ]
    )

    assert [item["id"] for item in report["missing_citations"]] == [
        "alpha-1",
        "alpha-2",
        "beta",
        "zeta",
    ]
    assert [item["id"] for item in report["results"]] == [
        "zeta",
        "cited",
        "alpha-2",
        "alpha-1",
        "beta",
    ]


def test_analyze_citation_coverage_is_importable_from_graph_rag():
    assert callable(analyze_citation_coverage)
