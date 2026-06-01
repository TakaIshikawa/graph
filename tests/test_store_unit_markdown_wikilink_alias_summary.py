from graph.store.unit_markdown_wikilink_alias_summary import summarize_unit_markdown_wikilink_aliases


def test_summarizes_aliased_wikilinks_and_ignores_plain_links():
    report = summarize_unit_markdown_wikilink_aliases([
        {"id": "b", "content": "[[Page|Alias]] and [[Plain]]\n```md\n[[Hidden|Alias]]\n```"},
        {"id": "a", "content": "[[Page|Alias]]\n[[Other#Heading|Different]]"},
    ])

    assert report["total_wikilinks"] == 4
    assert report["aliased_wikilinks"] == 3
    assert report["units_with_aliases"] == 2
    assert report["alias_target_pairs"] == [
        {"alias": "Alias", "target": "Page", "count": 2},
        {"alias": "Different", "target": "Other#Heading", "count": 1},
    ]
    assert report["top_aliases"] == [{"alias": "Alias", "count": 2}, {"alias": "Different", "count": 1}]
    assert report["samples"] == [
        {"unit_id": "a", "target": "Page", "alias": "Alias", "line_number": 1},
        {"unit_id": "a", "target": "Other#Heading", "alias": "Different", "line_number": 2},
        {"unit_id": "b", "target": "Page", "alias": "Alias", "line_number": 1},
    ]
