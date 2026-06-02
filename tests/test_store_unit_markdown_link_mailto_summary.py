from graph.store.unit_markdown_link_mailto_summary import summarize_unit_markdown_link_mailtos


def test_summarizes_mailto_links_query_strings_duplicates_and_samples():
    result = summarize_unit_markdown_link_mailtos(
        [
            {"id": "b", "content": "[Web](https://example.test) [B](mailto:b@example.test?subject=Hi) [B2](mailto:B@example.test)"},
            {"id": "a", "content": "[A](mailto:a@example.test)\n```\n[Skip](mailto:z@example.test)\n```"},
        ],
        sample_limit=1,
    )

    assert result == {
        "total_units": 2,
        "units_with_mailto_links": 2,
        "mailto_link_count": 3,
        "unique_email_count": 2,
        "samples": ["a@example.test"],
        "units": [
            {"unit_id": "a", "mailto_link_count": 1, "unique_email_count": 1, "sample_emails": ["a@example.test"]},
            {"unit_id": "b", "mailto_link_count": 2, "unique_email_count": 1, "sample_emails": ["b@example.test"]},
        ],
    }
