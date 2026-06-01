from graph.rag.result_duplicate_title_signal import analyze_result_duplicate_title_signals


def test_duplicate_titles_ignore_case_punctuation_and_whitespace():
    summary = analyze_result_duplicate_title_signals(
        [{"id": "a", "title": "Quarterly Report!"}, {"id": "b", "title": " quarterly   report "}]
    )

    assert summary["duplicate_group_count"] == 1
    assert summary["duplicate_titles"] == ["Quarterly Report!"]
    assert [member["result_id"] for member in summary["duplicate_results"][0]["members"]] == ["a", "b"]


def test_unique_or_missing_titles_do_not_create_duplicates():
    assert analyze_result_duplicate_title_signals([{"id": "a", "title": "A"}, {"id": "b"}, {"id": "c", "title": "C"}]) == {
        "duplicate_group_count": 0,
        "duplicate_titles": [],
        "duplicate_results": [],
    }
