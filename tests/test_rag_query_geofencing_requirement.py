from graph.rag.query_geofencing_requirement import detect_query_geofencing_requirement


def test_allow_only_region_is_allowlist():
    report = detect_query_geofencing_requirement("Allow only EU users with regional access rules.")

    assert report == {
        "requires_geofencing": True,
        "regions": ["eu"],
        "restriction_type": "allowlist",
        "matched_cues": ["location_access_restriction", "geofencing"],
        "severity": "high",
    }


def test_block_by_country_is_blocklist():
    report = detect_query_geofencing_requirement("Block users from Canada by IP geography.")

    assert report["requires_geofencing"] is True
    assert report["regions"] == ["canada"]
    assert report["restriction_type"] == "blocklist"
    assert report["severity"] == "high"


def test_plain_geographic_research_does_not_trigger():
    assert detect_query_geofencing_requirement("Compare ecommerce adoption in Europe and Japan.") == {
        "requires_geofencing": False,
        "regions": [],
        "restriction_type": "none",
        "matched_cues": [],
        "severity": "none",
    }
