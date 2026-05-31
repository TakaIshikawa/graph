from graph.store import summarize_unit_attachment_orphan_files


def test_attachment_orphan_file_summary_reports_orphans_and_missing_refs():
    units = [{"id": "u", "content": "![A](a.png)\n![[missing.pdf]]", "metadata": {"attachments": ["meta.txt"]}}]

    result = summarize_unit_attachment_orphan_files(units, ["a.png", "orphan.jpg"])

    assert result["referenced_count"] == 3
    assert result["available_count"] == 2
    assert result["orphan_files"] == ["orphan.jpg"]
    assert result["missing_references"] == ["meta.txt", "missing.pdf"]
