from __future__ import annotations

from graph.rag.query_sbom_requirement import detect_query_sbom_requirements


def test_detects_sbom_format_specific_signals():
    rows = detect_query_sbom_requirements("Require an SBOM in SPDX or CycloneDX format.")

    assert [row["category"] for row in rows] == ["sbom", "spdx", "cyclonedx"]


def test_detects_inventory_provenance_and_transitive_visibility():
    rows = detect_query_sbom_requirements(
        "Show component inventory, package provenance, and transitive dependency visibility."
    )

    assert [row["category"] for row in rows] == [
        "component_inventory",
        "package_provenance",
        "transitive_dependency_visibility",
    ]


def test_generic_dependency_question_without_sbom_intent_does_not_match():
    assert detect_query_sbom_requirements("Which dependency should we upgrade first?") == []
