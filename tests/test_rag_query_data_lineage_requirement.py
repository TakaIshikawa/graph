from graph.rag.query_data_lineage_requirement import detect_query_data_lineage_requirement


def test_data_lineage_and_traceability_trigger():
    report = detect_query_data_lineage_requirement("Need data lineage and source-to-output traceability.")

    assert report == {
        "requires_data_lineage": True,
        "lineage_scopes": ["data_lineage", "source_to_output"],
        "matched_cues": ["data_lineage", "source_to_output"],
        "confidence": "high",
    }


def test_upstream_and_downstream_scopes_are_reflected():
    report = detect_query_data_lineage_requirement(
        "Map upstream dependencies, downstream consumers, and transformation history."
    )

    assert report["lineage_scopes"] == ["upstream", "downstream", "transformation_history"]
    assert report["confidence"] == "medium"


def test_plain_source_citation_request_does_not_trigger():
    assert detect_query_data_lineage_requirement("Cite the sources used in the summary.") == {
        "requires_data_lineage": False,
        "lineage_scopes": [],
        "matched_cues": [],
        "confidence": "none",
    }
