from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_content_format_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, content: str = "", *, metadata: dict | None = None, source_project: object = "Source", entity_type: str = "note") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=f"Title {unit_id}",
        content=content,
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_content_format_classifies_common_shapes():
    text = export_source_content_format_csv(
        [
            unit("attachment", "", metadata={"attachments": ["file.pdf"]}),
            unit("empty", ""),
            unit("html", "<p>Hello</p>"),
            unit("json", '{"a": 1}'),
            unit("markdown", "# Heading"),
            unit("plain", "hello world"),
            unit("url", "https://example.com/path"),
        ]
    )

    assert [(row["content_format"], row["unit_count"], row["representative_unit_ids"]) for row in rows(text)] == [
        ("binary_attachment_reference", "1", "attachment"),
        ("empty", "1", "empty"),
        ("html_like", "1", "html"),
        ("json_like", "1", "json"),
        ("markdown_like", "1", "markdown"),
        ("plain_text", "1", "plain"),
        ("url_only", "1", "url"),
    ]


def test_source_content_format_groups_and_averages_chars():
    text = export_source_content_format_csv([unit("a", "aa"), unit("b", "bbbb")])

    assert rows(text) == [
        {
            "source_project": "Source",
            "source_entity_type": "note",
            "content_format": "plain_text",
            "unit_count": "2",
            "average_content_chars": "3.00",
            "representative_unit_ids": "a; b",
        }
    ]


def test_source_content_format_supports_mapping_inputs_unknowns_and_path(tmp_path):
    units = [{"id": "a", "content": "[]", "metadata": {}}]
    expected = export_source_content_format_csv(units)
    path = tmp_path / "formats.csv"
    stats = export_source_content_format_csv(units, path)

    assert rows(expected)[0]["source_project"] == "Unknown"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
