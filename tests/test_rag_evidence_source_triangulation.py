from __future__ import annotations

from types import SimpleNamespace

from graph.rag.evidence_source_triangulation import score_evidence_source_triangulation


def test_identifies_claims_supported_by_multiple_sources():
    report = score_evidence_source_triangulation(
        [
            {"id": "a", "text": "Revenue rose in 2025.", "domain": "one.test"},
            {"id": "b", "text": "Revenue rose in 2025.", "domain": "two.test"},
        ]
    )

    assert report["triangulated_claims"] == 1
    assert report["single_source_claims"] == 0
    assert report["distinct_sources"] == 2


def test_flags_claims_with_too_few_distinct_sources():
    report = score_evidence_source_triangulation([{"id": "a", "text": "Revenue rose in 2025.", "source": "Report A"}])

    assert report["single_source_claims"] == 1
    assert report["reason_counts"]["insufficient_distinct_sources"] == 1


def test_reads_source_identity_from_object_tuple_and_metadata():
    report = score_evidence_source_triangulation(
        [
            (SimpleNamespace(id="obj", text="The claim repeats.", metadata={"publisher": "Pub"}), 0.8),
            {"id": "url", "text": "The claim repeats.", "url": "https://example.test/a"},
        ]
    )

    assert report["distinct_sources"] == 2
    assert sorted(row["result_id"] for row in report["results"]) == ["obj", "url"]
