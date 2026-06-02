from graph.store import summarize_source_json_ld


def test_json_ld_summary_parses_objects_lists_html_and_invalid_payloads():
    summary = summarize_source_json_ld(
        [
            {"id": "a", "metadata": {"json_ld": {"@type": "Article"}}},
            {"id": "b", "metadata": {"html": '<script type="application/ld+json">[{"@type":"NewsArticle"},{"@type":["Thing","CreativeWork"]}]</script>'}},
            {"id": "c", "metadata": {"json_ld": "{broken"}},
            {"id": "d"},
        ]
    )

    assert summary["sources_with_json_ld"] == 2
    assert summary["invalid_json_ld_count"] == 1
    assert summary["missing_json_ld_count"] == 2
    assert summary["type_counts"] == {"Article": 1, "CreativeWork": 1, "NewsArticle": 1, "Thing": 1}
