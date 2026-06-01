from __future__ import annotations

from graph.rag.query_interoperability_requirement import detect_query_interoperability_requirements


def test_detect_query_interoperability_requirements_multiple_categories():
    rows = detect_query_interoperability_requirements(
        "Prefer open standards, data portability, backward-compatible APIs, vendor-neutral choices, and cross-platform clients."
    )

    assert [row["category"] for row in rows] == [
        "open_standards",
        "portability",
        "backward_compatibility",
        "vendor_neutrality",
        "cross_platform",
    ]


def test_detect_query_interoperability_requirements_compatibility_variants_and_exportability():
    rows = detect_query_interoperability_requirements("Need backwards compatible changes and exportable data.")

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("backward_compatibility", "backwards compatible"),
        ("data_exportability", "exportable data"),
    ]


def test_detect_query_interoperability_requirements_deduplicates_and_sorts():
    rows = detect_query_interoperability_requirements("Cross platform and cross-platform with no vendor lock-in.")

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("cross_platform", "cross platform"),
        ("vendor_neutrality", "no vendor lock-in"),
    ]


def test_detect_query_interoperability_requirements_no_match():
    assert detect_query_interoperability_requirements("Summarize onboarding steps.") == []
