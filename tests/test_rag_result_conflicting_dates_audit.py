from graph.rag.result_conflicting_dates_audit import audit_result_conflicting_dates


def test_parses_iso_dates_and_plain_years():
    result = audit_result_conflicting_dates([{"published_at": "2024-02-03", "snippet": "The older 2021 method is mentioned."}])

    assert result["date_field_counts"] == {"content_year": 1, "published_at": 1}
    assert result["conflicts"][0]["conflict_type"] == "conflicting_years"


def test_flags_updated_before_published_metadata_conflict():
    result = audit_result_conflicting_dates([{"published_at": "2024-04-01", "updated_at": "2024-03-01"}])

    assert result["affected_result_count"] == 1
    assert result["examples"] == [{"result_index": 0, "conflict_type": "updated_before_published", "first_value": "2024-04-01", "second_value": "2024-03-01"}]


def test_examples_are_deterministic_with_result_index_and_type():
    result = audit_result_conflicting_dates([
        {"year": "2020", "snippet": "Updated in 2023."},
        {"published": "2022-01-01", "updated": "2021-01-01"},
    ])

    assert [(item["result_index"], item["conflict_type"]) for item in result["examples"]] == [(0, "conflicting_years"), (1, "updated_before_published")]
