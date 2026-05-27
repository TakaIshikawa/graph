from graph.rag.context_persona_fit import audit_context_persona_fit


def test_infers_persona_cues_from_query():
    report = audit_context_persona_fit("Explain this for beginner engineers.", [])

    assert report["inferred_personas"] == ["beginner", "engineer"]


def test_counts_matching_mismatching_and_neutral_items():
    report = audit_context_persona_fit(
        "Brief this for executives.",
        [
            {"id": "a", "title": "Executive strategy guide"},
            {"id": "b", "metadata": {"audience": "developer API reference"}},
            {"id": "c", "title": "Market note"},
        ],
    )

    assert report["fit_counts"] == {"matching": 1, "mismatching": 1, "neutral": 1}
    assert report["mismatched_items"][0]["item_id"] == "b"


def test_no_query_persona_makes_items_neutral():
    report = audit_context_persona_fit("Summarize the topic.", [{"title": "Clinical guide"}])

    assert report["has_persona_mismatch_risk"] is False
    assert report["fit_counts"]["neutral"] == 1
