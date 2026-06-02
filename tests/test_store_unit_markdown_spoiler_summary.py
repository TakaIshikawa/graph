from graph.store.unit_markdown_spoiler_summary import summarize_unit_markdown_spoilers


def test_summarizes_pipe_spoilers_and_details_ignoring_fences():
    result = summarize_unit_markdown_spoilers(
        [
            {"id": "b", "content": "Plain"},
            {"id": "a", "content": "One ||secret|| and ||other||\n<details><summary>S</summary>\n```\n||skip||\n<details>\n```"},
        ]
    )

    assert result == {
        "total_units": 2,
        "units_with_spoilers": 1,
        "spoiler_count": 3,
        "details_count": 1,
        "pipe_spoiler_count": 2,
        "units": [{"unit_id": "a", "pipe_spoiler_count": 2, "details_count": 1, "spoiler_count": 3}],
    }
