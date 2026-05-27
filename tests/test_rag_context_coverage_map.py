from __future__ import annotations

from graph.rag.context_coverage_map import build_context_coverage_map


def test_context_coverage_map_matches_string_context_case_insensitively():
    report = build_context_coverage_map(
        ["Battery", "storage", "POLICY"],
        ["Battery storage planning memo.", "No policy details here."],
    )

    assert report == {
        "total_terms": 3,
        "covered_terms": ["battery", "storage", "policy"],
        "uncovered_terms": [],
        "term_coverage_ratio": 1.0,
        "item_coverage": [
            {
                "item_id": "context-1",
                "source": None,
                "matched_terms": ["battery", "storage"],
                "coverage_ratio": 0.6667,
            },
            {
                "item_id": "context-2",
                "source": None,
                "matched_terms": ["policy"],
                "coverage_ratio": 0.3333,
            },
        ],
        "coverage_flags": ["full_term_coverage"],
    }


def test_context_coverage_map_reads_mapping_text_content_snippet_and_ids():
    report = build_context_coverage_map(
        ["solar", "finance", "permitting"],
        [
            {"id": "a", "source": "docs", "text": "Solar deployment guide."},
            {"id": "b", "source": "reports", "content": "Finance model update."},
            {"id": "c", "source": "notes", "snippet": "Permitting checklist."},
        ],
    )

    assert report["covered_terms"] == ["solar", "finance", "permitting"]
    assert report["uncovered_terms"] == []
    assert report["item_coverage"] == [
        {
            "item_id": "a",
            "source": "docs",
            "matched_terms": ["solar"],
            "coverage_ratio": 0.3333,
        },
        {
            "item_id": "b",
            "source": "reports",
            "matched_terms": ["finance"],
            "coverage_ratio": 0.3333,
        },
        {
            "item_id": "c",
            "source": "notes",
            "matched_terms": ["permitting"],
            "coverage_ratio": 0.3333,
        },
    ]


def test_context_coverage_map_reports_uncovered_terms_and_flags():
    report = build_context_coverage_map(
        ["solar", "storage", "tariff"],
        [{"id": "r1", "content": "Solar storage costs fell."}],
    )

    assert report["total_terms"] == 3
    assert report["covered_terms"] == ["solar", "storage"]
    assert report["uncovered_terms"] == ["tariff"]
    assert report["term_coverage_ratio"] == 0.6667
    assert report["coverage_flags"] == ["partial_term_coverage"]


def test_context_coverage_map_deduplicates_repeated_terms_in_input_order():
    report = build_context_coverage_map(
        ["Policy", "battery", "policy", "Battery", "grid-scale"],
        [{"source_id": "src-1", "metadata": {"source": "archive"}, "text": "Battery policy memo."}],
    )

    assert report["total_terms"] == 3
    assert report["covered_terms"] == ["policy", "battery"]
    assert report["uncovered_terms"] == ["grid scale"]
    assert report["item_coverage"] == [
        {
            "item_id": "src-1",
            "source": "archive",
            "matched_terms": ["policy", "battery"],
            "coverage_ratio": 0.6667,
        }
    ]


def test_context_coverage_map_preserves_deterministic_item_ordering():
    report = build_context_coverage_map(
        ["alpha", "beta"],
        [
            {"id": "z", "text": "beta"},
            {"id": "a", "text": "alpha beta"},
            "alpha",
        ],
    )

    assert [row["item_id"] for row in report["item_coverage"]] == ["z", "a", "context-3"]
    assert [row["matched_terms"] for row in report["item_coverage"]] == [
        ["beta"],
        ["alpha", "beta"],
        ["alpha"],
    ]
