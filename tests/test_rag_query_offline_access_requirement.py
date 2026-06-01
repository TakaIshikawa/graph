from graph.rag.query_offline_access_requirement import detect_query_offline_access_requirement


def test_detects_offline_access_with_normalized_whitespace():
    result = detect_query_offline_access_requirement("Need   offline\noperation for field teams.")

    assert result == {
        "requires_offline_access": True,
        "signals": ["offline"],
        "strict_offline_required": False,
    }


def test_detects_strict_offline_cues_in_deterministic_order():
    result = detect_query_offline_access_requirement("Air gapped local-only deployment with no internet.")

    assert result["signals"] == ["air_gapped", "local_only", "no_internet"]
    assert result["strict_offline_required"] is True


def test_empty_and_unrelated_queries_do_not_match():
    assert detect_query_offline_access_requirement("") == {
        "requires_offline_access": False,
        "signals": [],
        "strict_offline_required": False,
    }
    assert detect_query_offline_access_requirement("Compare cloud hosted options.")["requires_offline_access"] is False
