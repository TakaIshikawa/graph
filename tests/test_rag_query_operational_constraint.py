from __future__ import annotations

from graph.rag.query_operational_constraint import detect_query_operational_constraint


def test_detects_multiple_operational_constraints_with_cues():
    report = detect_query_operational_constraint(
        "Recommend a rollout with no downtime, only 2 engineers, offline mode, "
        "a migration window, rollback plan, and no new dependencies."
    )

    assert report["has_operational_constraints"] is True
    assert report["constraint_types"] == [
        "dependency_freeze",
        "limited_staff",
        "migration_window",
        "no_downtime",
        "offline_mode",
        "rollback_requirement",
    ]
    assert [row["type"] for row in report["matched_cues"]] == [
        "no_downtime",
        "limited_staff",
        "offline_mode",
        "migration_window",
        "rollback_requirement",
        "dependency_freeze",
    ]
    assert [row["cue"] for row in report["matched_cues"]] == [
        "no downtime",
        "only 2 engineers",
        "offline mode",
        "migration window",
        "rollback plan",
        "no new dependencies",
    ]


def test_detects_maintenance_window_and_dependency_freeze_variants():
    report = detect_query_operational_constraint(
        "Plan this during scheduled maintenance; we cannot upgrade packages and need a backout procedure."
    )

    assert report["constraint_types"] == ["dependency_freeze", "maintenance_window", "rollback_requirement"]
    assert [row["cue"] for row in report["matched_cues"]] == [
        "scheduled maintenance",
        "cannot upgrade packages",
        "backout procedure",
    ]


def test_ordinary_informational_query_is_neutral():
    report = detect_query_operational_constraint("Summarize the recommended cache configuration options.")

    assert report == {"has_operational_constraints": False, "constraint_types": [], "matched_cues": []}
