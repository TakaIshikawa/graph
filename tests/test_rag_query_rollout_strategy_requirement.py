from __future__ import annotations

from graph.rag.query_rollout_strategy_requirement import detect_query_rollout_strategy_requirements


def test_detect_query_rollout_strategy_requirements_multiple_requirements():
    rows = detect_query_rollout_strategy_requirements(
        "Start with a pilot, then phased rollout using canary, blue-green, feature flags, and a beta."
    )

    assert [row["category"] for row in rows] == ["pilot", "phased", "canary", "blue_green", "feature_flag", "beta"]


def test_detect_query_rollout_strategy_requirements_overlapping_phrases_deduplicate():
    rows = detect_query_rollout_strategy_requirements("Use feature flag toggles for a canary rollout and canary release.")

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("feature_flag", "feature flag"),
        ("canary", "canary rollout"),
    ]


def test_detect_query_rollout_strategy_requirements_stable_sorting():
    rows = detect_query_rollout_strategy_requirements("Need an adoption plan after the migration window and private beta.")

    assert [row["category"] for row in rows] == ["adoption_plan", "migration_window", "beta"]
