from graph.rag.context_language_mix import analyze_context_language_mix


def test_reads_top_level_language_fields():
    report = analyze_context_language_mix([{"language": "EN"}, {"lang": "fr"}])

    assert report["language_counts"] == {"en": 1, "fr": 1}
    assert report["mixed_language"] is True


def test_reads_metadata_language_fields_and_missing_values():
    report = analyze_context_language_mix([{"metadata": {"language": "JA"}}, {}])

    assert report["language_counts"] == {"ja": 1}
    assert report["missing_language_count"] == 1


def test_dominant_language_ties_are_deterministic():
    report = analyze_context_language_mix([{"language": "fr"}, {"language": "en"}])

    assert report["dominant_language"] == "en"
