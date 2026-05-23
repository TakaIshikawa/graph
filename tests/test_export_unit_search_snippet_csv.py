from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_search_snippet_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_search_snippet_uses_title_tags_and_content_with_truncation():
    text = export_unit_search_snippet_csv(
        [{"id": "a", "title": "Alpha", "tags": ["z", "a"], "content": "First\n\ncontent excerpt"}],
        max_length=20,
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "tags": "a; z", "snippet": "Alpha a z First c...", "snippet_length": "20"}
    ]


def test_unit_search_snippet_path_mode_and_validation(tmp_path):
    path = tmp_path / "snippets.csv"
    stats = export_unit_search_snippet_csv([{"id": "a", "title": "", "tags": [], "content": "Body"}], path, max_length=4)

    assert rows(path.read_text(encoding="utf-8"))[0]["snippet"] == "Body"
    assert stats["max_length"] == 4
    with pytest.raises(ValueError, match="max_length"):
        export_unit_search_snippet_csv([], max_length=0)
