from graph.rag.query_data_minimization_requirement import detect_query_data_minimization_requirement


def test_explicit_pii_and_collect_only_necessary_are_high_severity():
    report = detect_query_data_minimization_requirement(
        "Collect only necessary data and minimize PII before retrieval."
    )

    assert report == {
        "requires_data_minimization": True,
        "matched_cues": ["collect_only_necessary", "pii_minimization"],
        "severity": "high",
    }


def test_detects_ordinary_minimization_cues():
    report = detect_query_data_minimization_requirement(
        "Use data minimization, truncate payloads, and anonymize records."
    )

    assert report == {
        "requires_data_minimization": True,
        "matched_cues": ["minimize_data", "truncate_data", "anonymize_data"],
        "severity": "medium",
    }


def test_no_match_defaults_are_safe():
    assert detect_query_data_minimization_requirement(None) == {
        "requires_data_minimization": False,
        "matched_cues": [],
        "severity": "none",
    }
    assert detect_query_data_minimization_requirement("Summarize model latency benchmarks.") == {
        "requires_data_minimization": False,
        "matched_cues": [],
        "severity": "none",
    }
