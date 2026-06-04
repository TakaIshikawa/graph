from graph.store import summarize_source_content_dispositions


def test_content_disposition_summary_counts_types_extensions_and_samples():
    summary = summarize_source_content_dispositions(
        [
            {"id": "a", "content_disposition": 'attachment; filename="report.PDF"'},
            {"id": "b", "metadata": {"Content-Disposition": "inline; filename*=UTF-8''preview%20image.png"}},
            {"id": "c", "headers": {"content-disposition": "attachment; filename=archive.tar.gz"}},
            {"id": "d"},
            {"id": "e", "response_headers": {"Content-Disposition": 'inline; filename="draft;v2.txt"'}},
        ]
    )

    assert summary["total_sources"] == 5
    assert summary["sources_with_content_disposition"] == 4
    assert summary["missing_content_disposition_count"] == 1
    assert summary["disposition_counts"] == {"attachment": 2, "inline": 2}
    assert summary["attachment_count"] == 2
    assert summary["inline_count"] == 2
    assert summary["filename_count"] == 4
    assert summary["filename_ext_counts"] == {".gz": 1, ".pdf": 1, ".png": 1, ".txt": 1}
    assert summary["malformed_count"] == 0
    assert summary["samples"] == [
        {"source_id": "a", "disposition": "attachment", "filename": "report.PDF", "raw": 'attachment; filename="report.PDF"'},
        {"source_id": "b", "disposition": "inline", "filename": "preview image.png", "raw": "inline; filename*=UTF-8''preview%20image.png"},
        {"source_id": "c", "disposition": "attachment", "filename": "archive.tar.gz", "raw": "attachment; filename=archive.tar.gz"},
        {"source_id": "e", "disposition": "inline", "filename": "draft;v2.txt", "raw": 'inline; filename="draft;v2.txt"'},
    ]


def test_content_disposition_summary_respects_sample_limit_and_header_case():
    summary = summarize_source_content_dispositions(
        [{"id": "a", "metadata": {"response_headers": {"CONTENT_DISPOSITION": "attachment; filename=x.txt"}}}],
        sample_limit=0,
    )

    assert summary["sources_with_content_disposition"] == 1
    assert summary["samples"] == []


def test_content_disposition_summary_counts_malformed_without_failing():
    summary = summarize_source_content_dispositions(
        [
            {"id": "a", "headers": {"content-disposition": "; filename=missing-type.txt"}},
            {"id": "b", "headers": {"content-disposition": "attachment; filename"}},
            {"id": "c", "headers": {"content-disposition": "inline; filename*=bad%ZZname.csv"}},
        ]
    )

    assert summary["sources_with_content_disposition"] == 3
    assert summary["disposition_counts"] == {"attachment": 1, "inline": 1, "unknown": 1}
    assert summary["filename_count"] == 2
    assert summary["filename_ext_counts"] == {".csv": 1, ".txt": 1}
    assert summary["malformed_count"] == 2
    assert summary["samples"][0] == {
        "source_id": "a",
        "disposition": "unknown",
        "filename": "missing-type.txt",
        "raw": "; filename=missing-type.txt",
    }
