from graph.store import summarize_source_subresource_integrity


def test_subresource_integrity_summary_detects_integrity_and_missing_values():
    summary = summarize_source_subresource_integrity(
        [
            {"id": "a", "content": '<script src="https://cdn.test/a.js" integrity="sha256-abc"></script>'},
            {"id": "b", "metadata": {"html": '<link href="https://cdn.test/b.css" integrity="sha384-def">'}},
            {"id": "c", "html": '<script src="https://cdn.test/c.js"></script><link href="/local.css">'},
            {"id": "d", "metadata": {"content": '<script src="https://cdn.test/d.js" integrity="md5-bad"></script>'}},
        ]
    )

    assert summary["integrity_algorithm_counts"] == {"sha256": 1, "sha384": 1}
    assert summary["missing_integrity_count"] == 1
    assert summary["invalid_integrity_count"] == 1
