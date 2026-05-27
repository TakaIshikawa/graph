from graph.rag.answer_scope_drift import audit_answer_scope_drift


def test_flags_excluded_topic_drift():
    report = audit_answer_scope_drift("Explain battery trends excluding lithium mining.", "It also includes lithium mining impacts.")

    assert report["has_scope_drift_risk"] is True
    assert report["drift_types"] == ["exclusion"]
    assert report["matched_cues"][0]["query_cue"] == "excluding lithium mining"


def test_flags_geography_drift():
    report = audit_answer_scope_drift("Summarize privacy rules in the EU.", "Worldwide, privacy rules vary by market.")

    assert report["drift_types"] == ["geographic"]


def test_flags_temporal_drift():
    report = audit_answer_scope_drift("Use evidence after 2022.", "Historically the pattern was different.")

    assert report["drift_types"] == ["temporal"]


def test_flags_source_constraint_drift():
    report = audit_answer_scope_drift("Use only official sources.", "Generally, blogs and forums agree.")

    assert report["drift_types"] == ["source"]


def test_compliant_answer_has_no_scope_drift():
    report = audit_answer_scope_drift("Use evidence after 2022 for the EU.", "In the EU after 2022, the rule changed.")

    assert report == {"has_scope_drift_risk": False, "drift_types": [], "matched_cues": []}
