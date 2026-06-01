from graph.store import summarize_source_duplicate_identifiers


def test_source_duplicate_identifier_summary_groups_duplicates_and_ignores_blanks():
    summary = summarize_source_duplicate_identifiers(
        [
            {"source_id": "a", "doi": "10/x"},
            {"source_id": "b", "metadata": {"doi": "10/x", "isbn": ""}},
            {"source_id": "c", "guid": "g"},
        ]
    )

    assert summary["duplicate_identifier_count"] == 1
    assert summary["duplicate_groups"] == [{"identifier_key": "doi", "identifier_value": "10/x", "source_count": 2, "source_ids": ["a", "b"]}]
    assert summary["key_counts"] == {"doi": 2, "guid": 1, "source_id": 3}


def test_source_duplicate_identifier_summary_custom_keys_override_defaults():
    summary = summarize_source_duplicate_identifiers([{"id": "a", "doi": "same"}, {"id": "b", "doi": "same"}], identifier_keys=["doi"])

    assert summary["key_counts"] == {"doi": 2}
    assert summary["duplicate_identifier_count"] == 1
