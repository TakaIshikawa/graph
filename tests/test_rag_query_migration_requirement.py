from __future__ import annotations

from graph.rag.query_migration_requirement import detect_query_migration_requirements


def test_detect_query_migration_requirements_migration_planning():
    rows = detect_query_migration_requirements("Plan the migration from the legacy platform with import and export steps.")

    assert [row["category"] for row in rows] == ["migration", "legacy", "import_export"]


def test_detect_query_migration_requirements_backfill_and_cutover():
    rows = detect_query_migration_requirements("Need a backfill plan for historical data before the cutover.")

    assert [row["category"] for row in rows] == ["backfill", "cutover"]


def test_detect_query_migration_requirements_portability():
    rows = detect_query_migration_requirements("Assess data portability and vendor exit options.")

    assert [row["category"] for row in rows] == ["portability"]


def test_detect_query_migration_requirements_ignores_unrelated_movement_language():
    assert detect_query_migration_requirements("Move the chart legend to the left side.") == []
