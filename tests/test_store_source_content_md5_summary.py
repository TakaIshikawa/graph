from graph.store import summarize_source_content_md5s


def test_content_md5_summary_counts_duplicates_invalid_and_missing():
    summary = summarize_source_content_md5s(
        [
            {"id": "a", "Content-MD5": "0123456789abcdef0123456789ABCDEF"},
            {"id": "b", "headers": {"content-md5": "0123456789abcdef0123456789abcdef"}},
            {"id": "c", "metadata": {"response_headers": {"CONTENT_MD5": "not-md5"}}},
            {"id": "d"},
        ]
    )

    assert summary["sources_with_content_md5"] == 3
    assert summary["duplicate_digest_count"] == 1
    assert summary["invalid_content_md5_count"] == 1
    assert summary["missing_content_md5_count"] == 1
    assert summary["samples"][2]["valid"] == "false"
