from graph.rag.answer_source_attribution_balance import audit_answer_source_attribution_balance


def test_balanced_attribution_counts_normalized_domains():
    summary = audit_answer_source_attribution_balance(
        "Nist.gov and example.org both describe the control.",
        [{"url": "https://www.NIST.gov/doc"}, {"domain": "example.org"}],
    )

    assert summary["evidence_source_count"] == 2
    assert summary["answer_source_mentions"] == {"example.org": 1, "nist.gov": 1}
    assert summary["single_source_overattribution"] is False


def test_single_source_overattribution_when_one_source_repeated():
    summary = audit_answer_source_attribution_balance(
        "Nist.gov says this. Nist.gov also recommends it.",
        [{"url": "https://nist.gov/a"}, {"url": "https://cisa.gov/b"}],
    )

    assert summary["dominant_answer_source"] == "nist.gov"
    assert summary["single_source_overattribution"] is True


def test_missing_metadata_returns_zero_sources():
    assert audit_answer_source_attribution_balance("No sources named.", [{}])["evidence_source_count"] == 0
