from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_code_pre_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_code_pre_csv_distinguishes_tags_language_and_fences():
    content = '<code class="language-python small">print()</code>\n<pre class="snippet">block</pre>\n```\n<code>skip</code>\n```'

    result = rows(export_unit_markdown_html_code_pre_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "tag": "code", "text": "print()", "class": "language-python small", "language_hint": "python"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "2", "tag": "pre", "text": "block", "class": "snippet", "language_hint": ""},
    ]
