from __future__ import annotations

from graph.rag.answer_citation_style_audit import audit_answer_citation_style


def test_answer_citation_style_audit_detects_styles_and_uncited_sentences():
    report = audit_answer_citation_style(
        "# Heading\nTrials improved outcomes [1]. See [CDC](https://cdc.gov). Smith reported replication (Smith, 2021). More detail.[^a]\nRaw source https://example.com. This factual sentence has no citation."
    )

    assert report["style_counts"] == {
        "numeric_bracket": 1,
        "markdown_link": 1,
        "author_year": 1,
        "footnote": 1,
        "bare_url": 2,
    }
    assert report["dominant_style"] == "bare_url"
    assert report["mixed_style_warning"] is True
    assert report["uncited_sentence_count"] == 1
    assert report["citation_style_score"] < 1
