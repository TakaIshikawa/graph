from graph.store.unit_markdown_bare_url_domain_summary import summarize_unit_markdown_bare_url_domains


def test_summarize_unit_markdown_bare_url_domains_counts_domains():
    report = summarize_unit_markdown_bare_url_domains([
        {"id": "b", "content": "Visit https://Example.com/a and http://sub.example.com:8080/path."},
        {"id": "a", "content": "Again https://example.com/next"},
    ])

    assert report["total_units"] == 2
    assert report["bare_url_count"] == 3
    assert report["domain_counts"] == {"example.com": 2, "sub.example.com": 1}
    assert report["affected_units"] == ["a", "b"]
    assert report["examples"][0] == {"unit_id": "a", "line_number": 1, "domain": "example.com", "url": "https://example.com/next"}


def test_summarize_unit_markdown_bare_url_domains_ignores_markdown_links_autolinks_and_fences():
    content = "\n".join([
        "[linked](https://linked.example/path)",
        "![image](https://image.example/img.png)",
        "<https://auto.example/path>",
        "bare https://bare.example/path)",
        "```",
        "https://fenced.example",
        "```",
    ])

    report = summarize_unit_markdown_bare_url_domains([{"id": "u", "content": content}])

    assert report["bare_url_count"] == 1
    assert report["domain_counts"] == {"bare.example": 1}
    assert report["examples"] == [{"unit_id": "u", "line_number": 4, "domain": "bare.example", "url": "https://bare.example/path"}]
