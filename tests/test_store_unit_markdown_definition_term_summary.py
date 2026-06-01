from graph.store.unit_markdown_definition_term_summary import summarize_unit_markdown_definition_terms


def test_summarize_unit_markdown_definition_terms_counts_terms_and_definitions():
    report = summarize_unit_markdown_definition_terms([
        {"id": "u", "content": "Term\n: one\n: two\n\nOther\n: single"},
    ])

    assert report["term_count"] == 2
    assert report["definition_count"] == 3
    assert report["multi_definition_term_count"] == 1
    assert report["orphan_definition_count"] == 0
    assert report["term_counts"] == {"Other": 1, "Term": 1}


def test_summarize_unit_markdown_definition_terms_counts_stacked_terms_and_orphans():
    report = summarize_unit_markdown_definition_terms([
        {"id": "u", "content": "Term A\nTerm B\n: shared\n\n: orphan\nhttp://x: no\n```\nHidden\n: no\n```"},
    ])

    assert report["total_units"] == 1
    assert report["term_count"] == 2
    assert report["definition_count"] == 2
    assert report["orphan_definition_count"] == 1
    assert report["multi_definition_term_count"] == 0
    assert report["term_counts"] == {"Term A": 1, "Term B": 1}
