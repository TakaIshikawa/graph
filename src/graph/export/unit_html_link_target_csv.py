"""CSV export for raw HTML anchor targets in unit fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "target", "target_type", "anchor_text", "source_field"]


def export_units_to_html_link_target_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        fields = [("content", get(unit, "content"))]
        fields.extend((f"metadata.{key}", value) for key, value in metadata(unit).items() if field_value(key) in {"html", "body"})
        for source, html in fields:
            for target, anchor in _anchors(str(html or "")):
                rows.append({"unit_id": unit_id(unit), "title": title, "target": target, "target_type": _target_type(target), "anchor_text": anchor, "source_field": source})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["target"]), sort_key(row["source_field"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "a":
            self._href = next((value for key, value in attrs if key.casefold() == "href"), None)
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._href is not None:
            self.rows.append((field_value(self._href), field_value(" ".join(self._text))))
            self._href = None
            self._text = []


def _anchors(html: str) -> list[tuple[str, str]]:
    parser = _AnchorParser()
    parser.feed(html)
    return [(target, text) for target, text in parser.rows if target]


def _target_type(target: str) -> str:
    parsed = urlparse(target)
    scheme = parsed.scheme.casefold()
    if target.startswith("#"):
        return "fragment"
    if scheme == "mailto":
        return "mailto"
    if scheme == "tel":
        return "tel"
    if scheme in {"http", "https"} and parsed.netloc:
        return "external"
    if not scheme and target:
        return "internal"
    return "invalid"
