from __future__ import annotations

from graph.rag.query_integration_surface_requirement import detect_query_integration_surface_requirements


def test_detect_query_integration_surface_requirements_multiple_surfaces():
    rows = detect_query_integration_surface_requirements(
        "Need REST API docs, webhooks, command line examples, SDK guidance, and an MCP server."
    )

    assert [(row["surface"], row["matched_text"], row["severity"]) for row in rows] == [
        ("api", "rest api", "high"),
        ("webhook", "webhooks", "high"),
        ("cli", "command line", "medium"),
        ("sdk", "sdk", "medium"),
        ("mcp_server", "mcp server", "high"),
    ]


def test_detect_query_integration_surface_requirements_file_and_database_surfaces():
    rows = detect_query_integration_surface_requirements(
        "Compare database connector support, file import for CSV, and export data options."
    )

    assert [row["surface"] for row in rows] == ["database_connector", "file_import", "file_export"]


def test_detect_query_integration_surface_requirements_deduplicates_by_surface_and_sorts_by_match_order():
    rows = detect_query_integration_surface_requirements("Webhooks plus REST API endpoints and another API.")

    assert [(row["surface"], row["matched_text"]) for row in rows] == [
        ("webhook", "webhooks"),
        ("api", "rest api"),
    ]


def test_detect_query_integration_surface_requirements_empty_query_returns_no_rows():
    assert detect_query_integration_surface_requirements("") == []
