from graph.store.unit_markdown_link_rel_attribute_summary import summarize_unit_markdown_link_rel_attributes


def test_link_rel_attribute_summary_extracts_values_case_insensitively():
    summary = summarize_unit_markdown_link_rel_attributes(
        [
            {"id": "b", "content": "<a href='x' REL=\"noopener noreferrer\">x</a> <a href='y'>y</a>"},
            {"id": "a", "content": "<a rel='nofollow ugc sponsored' href='z'>z</a>\n```\n<a rel='skip'>\n```"},
        ],
        sample_limit=4,
    )

    assert summary["total_units"] == 2
    assert summary["anchor_count"] == 3
    assert summary["missing_rel_anchor_count"] == 1
    assert summary["affected_units"] == 2
    assert summary["rel_value_counts"] == {"nofollow": 1, "noopener": 1, "noreferrer": 1, "sponsored": 1, "ugc": 1}
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "rel": "nofollow"},
        {"unit_id": "a", "line": 1, "rel": "sponsored"},
        {"unit_id": "a", "line": 1, "rel": "ugc"},
        {"unit_id": "b", "line": 1, "rel": ""},
    ]
