from __future__ import annotations

from graph.store import summarize_unit_local_file_references


def test_summarize_unit_local_file_references_groups_extension_scheme_and_style():
    summary = summarize_unit_local_file_references([{"id": "a", "content": "[doc](docs/a.md)\n[file](file:///tmp/b.pdf)\n[web](https://example.com)"}, {"id": "b", "content": "plain"}])

    assert summary["local_reference_count"] == 2
    assert summary["extensions"] == [{"extension": "md", "count": 1}, {"extension": "pdf", "count": 1}]
    assert summary["schemes"] == [{"scheme": "(none)", "count": 1}, {"scheme": "file", "count": 1}]
    assert summary["path_styles"] == [{"path_style": "absolute", "count": 1}, {"path_style": "relative", "count": 1}]
