from graph.rag.result_canonical_url import audit_result_canonical_urls


def test_groups_duplicate_urls_after_normalization():
    report = audit_result_canonical_urls(
        [{"id": "a", "url": "HTTPS://Example.com/Page/"}, {"id": "b", "url": "https://example.com/Page#section"}]
    )

    assert report["duplicate_url_groups"][0]["canonical_url"] == "https://example.com/Page"
    assert [item["item_id"] for item in report["duplicate_url_groups"][0]["items"]] == ["a", "b"]


def test_reports_missing_urls_with_stable_ids():
    report = audit_result_canonical_urls([{"id": "a"}, {}])

    assert report["missing_url_items"] == [{"item_id": "a", "index": 0}, {"item_id": "result-2", "index": 1}]


def test_reads_metadata_urls_and_strips_fragments():
    report = audit_result_canonical_urls([{"metadata": {"source_url": "https://x.test/a#b"}}])

    assert report["unique_url_count"] == 1
    assert report["duplicate_url_groups"] == []
