from __future__ import annotations

from graph.rag.query_data_format_requirement import detect_query_data_format_requirement


def test_detects_common_formats_case_insensitively():
    report = detect_query_data_format_requirement(
        "Return CSV, JSON, XML, YAML, SQL, Parquet, and Excel sources."
    )

    assert report["requires_structured_data"] is True
    assert report["requested_formats"] == ["csv", "json", "xml", "yaml", "sql", "parquet", "spreadsheet"]
    assert report["schema_cues"] == []
    assert report["machine_readable_cues"] == []
    assert report["confidence"] == 0.85
    assert report["recommendations"] == [
        "prefer_sources_with_structured_or_exportable_data",
        "preserve_field_names_types_and_units_from_source",
    ]


def test_distinguishes_schema_and_api_cues_from_generic_table_mentions():
    report = detect_query_data_format_requirement(
        "Find the OpenAPI schema and API response body, plus a table of fields."
    )

    assert report["requested_formats"] == ["api_response", "schema", "table"]
    assert [cue["type"] for cue in report["schema_cues"]] == ["schema", "api_response"]
    assert [cue["cue"] for cue in report["schema_cues"]] == ["openapi schema", "api response"]
    assert report["machine_readable_cues"] == []
    assert report["recommendations"] == [
        "prefer_sources_with_structured_or_exportable_data",
        "retrieve_schema_api_or_contract_documentation",
    ]
    assert report["confidence"] == 0.75


def test_machine_readable_cues_are_reported_separately():
    report = detect_query_data_format_requirement("Need machine-readable structured data, preferably JSONL.")

    assert report["requested_formats"] == ["json", "machine_readable"]
    assert [cue["cue"] for cue in report["machine_readable_cues"]] == [
        "machine-readable",
        "structured data",
    ]
    assert report["recommendations"] == [
        "prefer_sources_with_structured_or_exportable_data",
        "prioritize_machine_readable_sources_over_narrative_summaries",
        "preserve_field_names_types_and_units_from_source",
    ]
    assert report["confidence"] == 0.95


def test_generic_table_only_has_low_confidence_without_schema_cue():
    report = detect_query_data_format_requirement("Show a table comparing the supported options.")

    assert report["requires_structured_data"] is True
    assert report["requested_formats"] == ["table"]
    assert report["schema_cues"] == []
    assert report["machine_readable_cues"] == []
    assert report["confidence"] == 0.45


def test_no_format_query_is_neutral():
    report = detect_query_data_format_requirement("Summarize the deployment recommendations.")

    assert report == {
        "requires_structured_data": False,
        "requested_formats": [],
        "schema_cues": [],
        "machine_readable_cues": [],
        "recommendations": [],
        "confidence": 0.0,
        "normalized_query": "summarize the deployment recommendations.",
    }
