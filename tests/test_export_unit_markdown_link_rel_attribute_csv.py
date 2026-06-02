import csv
from io import StringIO

from graph.export import export_units_to_markdown_link_rel_attribute_csv


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_markdown_link_rel_attribute_csv_reports_markdown_and_html_rel_tokens():
    content = "\n".join(
        [
            "[Docs](https://a.test){rel=\"NoFollow noopener\"}",
            "<a href='/b' rel='noreferrer tag'>Beta</a>",
            "[No rel](https://c.test){title=\"Only title\"}",
            "```",
            "[Ignored](https://d.test){rel=\"nofollow\"}",
            "```",
        ]
    )

    rows = _rows(export_units_to_markdown_link_rel_attribute_csv([{"id": "u1", "title": "Links", "content": content}]))

    assert rows == [
        {"unit_id": "u1", "title": "Links", "link_text": "Docs", "href": "https://a.test", "rel_value": "NoFollow noopener", "rel_token_count": "2", "nofollow": "True", "noopener": "True", "noreferrer": "False", "line_number": "1"},
        {"unit_id": "u1", "title": "Links", "link_text": "Beta", "href": "/b", "rel_value": "noreferrer tag", "rel_token_count": "2", "nofollow": "False", "noopener": "False", "noreferrer": "True", "line_number": "2"},
    ]


def test_markdown_link_rel_attribute_csv_multiple_units_sort_and_path_stats(tmp_path):
    path = tmp_path / "link-rel.csv"
    units = [{"id": "b", "content": "<a href=\"/b\" rel=\"noopener\">B</a>"}, {"id": "a", "content": "[A](/a){rel=\"nofollow noreferrer\"}"}]

    expected = export_units_to_markdown_link_rel_attribute_csv(units)
    stats = export_units_to_markdown_link_rel_attribute_csv(units, path)

    assert [row["unit_id"] for row in _rows(expected)] == ["a", "b"]
    assert stats["rows_exported"] == 2
    assert path.read_text(encoding="utf-8") == expected
