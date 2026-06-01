from graph.store import summarize_source_charsets


def test_source_charset_summary_normalizes_and_samples_non_utf8():
    summary = summarize_source_charsets(
        [
            {"source_id": "a", "charset": "utf8"},
            {"source_id": "b", "metadata": {"content_type": "text/html; charset=Shift_JIS"}},
            {"source_id": "c", "headers": {"Content-Type": "application/json; charset=UTF-8"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_charset"] == 3
    assert summary["charset_counts"] == {"shift_jis": 1, "utf-8": 2}
    assert summary["missing_charset_count"] == 1
    assert summary["non_utf8_count"] == 1
    assert summary["samples"] == [{"source_id": "b", "charset": "shift_jis"}]
