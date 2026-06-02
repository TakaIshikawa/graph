from __future__ import annotations

import pytest

from graph.rag.query_migration_plan_requirement import detect_query_migration_plan_requirement


def test_detects_migration_plan_cues_and_phase_terms():
    result = detect_query_migration_plan_requirement(
        "Need a migration plan for pilot, phase 1, wave 2, weekend cutover, and rollback window."
    )

    assert result["requires_migration_plan"] is True
    assert result["cue_categories"] == ["migration_plan", "cutover"]
    assert result["phase_terms"] == ["pilot", "phase 1", "wave 2", "weekend cutover", "rollback window"]


def test_detects_cutover_rollback_parallel_and_go_live_cues():
    result = detect_query_migration_plan_requirement(
        "Provide phased migration, data migration, parallel run, backout, rollback plan, and go-live checklist evidence."
    )

    assert result["requires_migration_plan"] is True
    assert result["cue_categories"] == [
        "phased_migration",
        "data_migration",
        "parallel_run",
        "backout",
        "rollback_plan",
        "go_live_checklist",
    ]


def test_generic_implementation_query_does_not_match():
    result = detect_query_migration_plan_requirement("How is the connector implemented?")

    assert result["requires_migration_plan"] is False
    assert result["cue_categories"] == []


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_migration_plan_requirement("")
