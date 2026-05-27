"""Summarize HTML heading anchors in unit content and metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id


def summarize_unit_html_heading_anchors(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    anchors: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    examples = []
    for unit in units:
        total_units += 1
        source = field_value(get(unit, "source") or metadata(unit).get("source"))
        fields = [get(unit, "content"), *[value for value in metadata(unit).values() if isinstance(value, str)]]
        for html in fields:
            for level, anchor_id, text in _anchors(str(html or "")):
                anchors[anchor_id] += 1
                levels[level] += 1
                if len(examples) < sample_limit:
                    examples.append({"unit_id": unit_id(unit), "source": source, "anchor_id": anchor_id, "heading_text": text})
    duplicate_anchor_ids = [{"anchor_id": anchor_id, "count": count} for anchor_id, count in anchors.items() if count > 1]
    duplicate_anchor_ids.sort(key=lambda row: (-row["count"], sort_key(row["anchor_id"])))
    return {
        "total_units": total_units,
        "total_anchors": sum(anchors.values()),
        "duplicate_anchor_ids": duplicate_anchor_ids,
        "anchors_by_level": dict(sorted(levels.items(), key=lambda item: sort_key(item[0]))),
        "examples": examples,
    }


class _HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, str]] = []
        self._active: tuple[str, str] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        level = tag.casefold()
        if level in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            anchor_id = next((value for key, value in attrs if key.casefold() == "id" and value), "")
            if anchor_id:
                self._active = (level, anchor_id)
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._active and tag.casefold() == self._active[0]:
            self.rows.append((self._active[0], field_value(self._active[1]), field_value(" ".join(self._text))))
            self._active = None
            self._text = []


def _anchors(html: str) -> list[tuple[str, str, str]]:
    parser = _HeadingParser()
    parser.feed(html)
    return parser.rows
