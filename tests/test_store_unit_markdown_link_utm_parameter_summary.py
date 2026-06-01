from graph.store.unit_markdown_link_utm_parameter_summary import summarize_unit_markdown_link_utm_parameters


def test_summarizes_utm_parameters_from_markdown_and_bare_urls():
    report = summarize_unit_markdown_link_utm_parameters([
        {"id": "b", "content": "[x](https://e.test/?UTM_Source=News&utm_medium=email&utm_campaign=a) https://e.test/plain"},
        {"id": "a", "content": "https://e.test/?utm_source=social&utm_content=post\n```\nhttps://e.test/?utm_source=hidden\n```"},
    ])

    assert report["utm_link_count"] == 2
    assert report["units_with_utm_links"] == 2
    assert report["parameter_name_counts"] == {"utm_campaign": 1, "utm_content": 1, "utm_medium": 1, "utm_source": 2}
    assert report["source_counts"] == {"News": 1, "social": 1}
    assert report["medium_counts"] == {"email": 1}
    assert report["samples"] == [
        {"unit_id": "a", "url": "https://e.test/?utm_source=social&utm_content=post", "parameter_names": ["utm_content", "utm_source"]},
        {"unit_id": "b", "url": "https://e.test/?UTM_Source=News&utm_medium=email&utm_campaign=a", "parameter_names": ["utm_campaign", "utm_medium", "utm_source"]},
    ]
