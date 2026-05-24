from __future__ import annotations

from graph.rag.evidence_methodology_completeness import audit_evidence_methodology_completeness


def test_scores_methodology_completeness_per_result():
    report = audit_evidence_methodology_completeness(
        [
            {
                "id": "r1",
                "snippet": "A sample of 500 adults was measured by survey during 2024; limitations include self-reported bias.",
            },
            {"id": "r2", "title": "Revenue increased"},
        ]
    )

    assert report["rows"][0]["completeness_score"] == 1.0
    assert report["rows"][1]["completeness_score"] == 0.0
    assert report["missing_signal_counts"]["sample"] == 1


def test_reads_text_content_snippet_and_title_fields():
    report = audit_evidence_methodology_completeness([{"title": "2025 cohort", "content": "patients endpoint"}])

    assert report["rows"][0]["signals_present"] == ["population", "measurement", "timeframe"]
