from __future__ import annotations

from graph.store.unit_markdown_reference_link_text_summary import summarize_unit_markdown_reference_link_texts


def test_reference_link_text_summary_groups_by_normalized_label_and_sorts():
    summary = summarize_unit_markdown_reference_link_texts(
        [
            {"id": "u1", "content": "[Docs][API Docs]\n[Guide][]\n[API Docs]: https://example.com"},
            {"id": "u2", "content": "[Other][api   docs] and [Guide][]"},
        ]
    )

    assert summary["total_units"] == 2
    assert summary["reference_labels"][0] == {
        "label": "api docs",
        "use_count": 2,
        "unit_count": 2,
        "link_texts": ["Docs", "Other"],
        "examples": [
            {"unit_id": "u1", "line_number": 1, "link_text": "Docs", "usage_type": "full"},
            {"unit_id": "u2", "line_number": 1, "link_text": "Other", "usage_type": "full"},
        ],
    }
    assert summary["reference_labels"][1]["label"] == "guide"


def test_reference_link_text_summary_counts_collapsed_shortcut_and_ignores_code():
    summary = summarize_unit_markdown_reference_link_texts(
        [
            {
                "id": "u1",
                "content": "`[Code][ref]` [Visible][] [Shortcut]\n```\n[Fenced][ref]\n```\n![Alt][image]",
            }
        ]
    )

    assert [(row["label"], row["use_count"]) for row in summary["reference_labels"]] == [("shortcut", 1), ("visible", 1)]
    assert summary["reference_labels"][0]["examples"][0]["usage_type"] == "shortcut"
