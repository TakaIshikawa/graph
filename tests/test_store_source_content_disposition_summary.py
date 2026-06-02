from graph.store import summarize_source_content_dispositions


def test_content_disposition_summary_counts_types_extensions_and_samples():
    summary = summarize_source_content_dispositions(
        [
            {"id": "a", "content_disposition": 'attachment; filename="report.PDF"'},
            {"id": "b", "metadata": {"Content-Disposition": "inline; filename*=UTF-8''preview%20image.png"}},
            {"id": "c", "headers": {"content-disposition": "attachment; filename=archive.tar.gz"}},
            {"id": "d"},
        ]
    )

    assert summary["sources_with_content_disposition"] == 3
    assert summary["missing_content_disposition_count"] == 1
    assert summary["disposition_type_counts"] == {"attachment": 2, "inline": 1}
    assert summary["filename_extension_counts"] == {".gz": 1, ".pdf": 1, ".png": 1}
    assert summary["samples"] == [
        {"source_id": "a", "disposition_type": "attachment", "filename": "report.PDF", "raw": 'attachment; filename="report.PDF"'},
        {"source_id": "b", "disposition_type": "inline", "filename": "preview image.png", "raw": "inline; filename*=UTF-8''preview%20image.png"},
        {"source_id": "c", "disposition_type": "attachment", "filename": "archive.tar.gz", "raw": "attachment; filename=archive.tar.gz"},
    ]


def test_content_disposition_summary_respects_sample_limit_and_header_case():
    summary = summarize_source_content_dispositions(
        [{"id": "a", "metadata": {"response_headers": {"CONTENT_DISPOSITION": "attachment; filename=x.txt"}}}],
        sample_limit=0,
    )

    assert summary["sources_with_content_disposition"] == 1
    assert summary["samples"] == []
