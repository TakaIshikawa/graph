from graph.rag.query_data_portability_requirement import detect_query_data_portability_requirement


def test_detect_query_data_portability_requirement_identifies_bulk_export():
    result = detect_query_data_portability_requirement("Need data portability terms and bulk export of customer data.")

    assert result["requires_data_portability"] is True
    assert result["categories"] == ["bulk_export", "data_portability"]
    assert [match["matched_text"] for match in result["matches"]] == ["data portability", "bulk export"]


def test_detect_query_data_portability_requirement_recognizes_machine_readable_offboarding():
    result = detect_query_data_portability_requirement("Can customers migrate off with machine-readable exports in CSV?")

    assert result["requires_data_portability"] is True
    assert result["categories"] == ["export_format", "machine_readable_export", "offboarding_migration"]


def test_detect_query_data_portability_requirement_ignores_unrelated_dashboard_export():
    assert detect_query_data_portability_requirement("Export the dashboard screenshot for the weekly reporting deck.") == {
        "requires_data_portability": False,
        "categories": [],
        "matches": [],
    }
