from graph.store.unit_markdown_callout_fold_state_summary import summarize_unit_markdown_callout_fold_states


def test_summarizes_callout_fold_states_and_types():
    result = summarize_unit_markdown_callout_fold_states(
        [
            {"id": "b", "content": "> quote\n> [!TIP]+ Open\n> [!warning]- Closed"},
            {"id": "a", "content": "> [!tip] Plain"},
        ]
    )

    assert result == {
        "total_units": 2,
        "units_with_callouts": 2,
        "callout_count": 3,
        "fold_states": {"open": 1, "closed": 1, "none": 1},
        "types": {"tip": 2, "warning": 1},
        "units": [
            {"unit_id": "a", "callout_count": 1, "open_count": 0, "closed_count": 0, "none_count": 1},
            {"unit_id": "b", "callout_count": 2, "open_count": 1, "closed_count": 1, "none_count": 0},
        ],
    }
