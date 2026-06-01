from graph.store.unit_markdown_reference_link_usage_summary import summarize_unit_markdown_reference_link_usage


def test_reference_link_usage_summary_counts_resolved_forms_and_excludes_images_fences():
    summary = summarize_unit_markdown_reference_link_usage(
        [
            {
                "id": "b",
                "content": "[Docs][docs] [Guide][] [Shortcut] ![Img][docs] [Missing]\n[docs]: /docs\n[guide]: /guide\n[shortcut]: /s\n```md\n[Docs][docs]\n```",
            },
            {"id": "a", "content": "[A][id]\n[id]: /a"},
        ],
        sample_limit=3,
    )

    assert summary["total_units"] == 2
    assert summary["units_with_reference_link_usage"] == 2
    assert summary["full_reference_count"] == 2
    assert summary["collapsed_reference_count"] == 1
    assert summary["shortcut_reference_count"] == 1
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "usage_type": "full", "label": "id"},
        {"unit_id": "b", "line": 1, "usage_type": "collapsed", "label": "Guide"},
        {"unit_id": "b", "line": 1, "usage_type": "full", "label": "docs"},
    ]
