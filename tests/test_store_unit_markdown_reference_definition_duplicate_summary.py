from graph.store.unit_markdown_reference_definition_duplicate_summary import summarize_unit_markdown_reference_definition_duplicates


def test_reference_definition_duplicate_summary_normalizes_labels_and_targets():
    summary = summarize_unit_markdown_reference_definition_duplicates(
        [
            {"id": "b", "content": "[A  B]: /first\n[a b]: /second\n[unique]: /u"},
            {"id": "a", "content": "[Docs]: https://one\n[docs]: https://two\n```md\n[docs]: skip\n```"},
        ],
        sample_limit=1,
    )

    assert summary["total_units"] == 2
    assert summary["duplicate_label_count"] == 2
    assert summary["affected_units"] == 2
    assert summary["examples"] == [
        {
            "unit_id": "a",
            "label": "docs",
            "occurrence_count": 2,
            "first_target": "https://one",
            "duplicate_targets": ["https://two"],
        }
    ]


def test_reference_definition_duplicate_summary_returns_all_examples_stably():
    summary = summarize_unit_markdown_reference_definition_duplicates(
        [{"id": "u", "content": "[x]: /1\n[X]: /2\n[y]: /3\n[Y]: /4"}],
        sample_limit=5,
    )

    assert [row["label"] for row in summary["examples"]] == ["x", "y"]
