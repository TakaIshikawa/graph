from __future__ import annotations

from graph.rag import detect_citation_gaps


def test_citation_gap_detector_reports_missing_required_citation_fields():
    gaps = detect_citation_gaps(
        [
            {
                "id": "partial",
                "title": "Partially cited result",
                "source": "Lab notes",
                "metadata": {"url": "https://example.test/result"},
            }
        ]
    )

    assert gaps == [
        {
            "result_id": "partial",
            "title": "Partially cited result",
            "missing_fields": ["author", "date"],
            "missing_count": 2,
            "severity": "low",
        }
    ]


def test_citation_gap_detector_omits_complete_citation_metadata():
    gaps = detect_citation_gaps(
        [
            {
                "id": "complete",
                "title": "Complete result",
                "metadata": {
                    "source": "Journal",
                    "url": "https://example.test/paper",
                    "author": "Ada Lovelace",
                    "publication_date": "2026-04-01",
                },
            },
            {
                "id": "missing",
                "title": "Missing URL",
                "source": "Archive",
                "author": "Grace Hopper",
                "date": "2026-04-02",
            },
        ]
    )

    assert [gap["result_id"] for gap in gaps] == ["missing"]
    assert gaps[0]["missing_fields"] == ["url"]
    assert gaps[0]["severity"] == "medium"


def test_citation_gap_detector_supports_outline_sections_and_nested_units():
    gaps = detect_citation_gaps(
        [
            {
                "section_id": "outline-1",
                "heading": "Answer section",
                "citation": {"source": "Report", "link": "https://example.test/report"},
                "metadata": {"authors": ["Smith"], "published_at": "2026-05-01"},
            },
            {
                "unit": {
                    "id": "unit-a",
                    "title": "Nested unit",
                    "metadata": {"source_name": "Notebook", "created_at": "2026-05-02"},
                }
            },
        ]
    )

    assert gaps == [
        {
            "result_id": "unit-a",
            "title": "Nested unit",
            "missing_fields": ["url", "author"],
            "missing_count": 2,
            "severity": "medium",
        }
    ]


def test_citation_gap_detector_empty_input_and_stable_sorting():
    assert detect_citation_gaps([]) == []

    results = [
        {"id": "b", "title": "Beta", "metadata": {}},
        {"id": "a", "title": "Alpha", "metadata": {"source": "Archive"}},
        {"id": "c", "title": "Complete", "metadata": {"source": "Archive", "url": "u", "author": "a", "date": "d"}},
    ]

    assert detect_citation_gaps(results) == detect_citation_gaps(reversed(results))
    assert [gap["result_id"] for gap in detect_citation_gaps(results)] == ["b", "a"]
    assert [gap["severity"] for gap in detect_citation_gaps(results)] == ["high", "high"]
