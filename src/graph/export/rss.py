"""RSS 2.0 feed export for knowledge units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timezone
from email.utils import format_datetime
from enum import Enum
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_URL_KEYS = ("url", "source_url", "external_url", "uri", "link")


def export_units_to_rss(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    channel_title: str = "Knowledge Graph",
    channel_link: str = "",
    channel_description: str = "Knowledge graph feed",
) -> str:
    """Export units as deterministic RSS 2.0 XML."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)

    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = channel_title
    SubElement(channel, "link").text = channel_link
    SubElement(channel, "description").text = channel_description
    SubElement(channel, "lastBuildDate").text = _rss_date(_latest_datetime(unit_list))

    for unit in unit_list:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = unit.title

        url = _unit_url(unit.metadata)
        guid_text = url or unit.source_id or unit.id or ""
        guid = SubElement(item, "guid")
        guid.text = guid_text
        if not url:
            guid.set("isPermaLink", "false")

        if url:
            SubElement(item, "link").text = url

        published = unit.created_at or unit.updated_at
        if published:
            SubElement(item, "pubDate").text = _rss_date(published)

        for tag in unit.tags:
            category = _clean_text(_scalar_text(tag))
            if category:
                SubElement(item, "category").text = category

        if unit.content:
            SubElement(item, "description").text = unit.content

    xml = tostring(rss, encoding="unicode", xml_declaration=True)
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml, encoding="utf-8")
    return xml


def _latest_datetime(units: list[KnowledgeUnit]) -> datetime:
    dates = [unit.updated_at or unit.created_at for unit in units if unit.updated_at or unit.created_at]
    if dates:
        return max(_aware_datetime(value) for value in dates)
    return datetime.fromtimestamp(0, timezone.utc)


def _rss_date(value: datetime | date) -> str:
    return format_datetime(_aware_datetime(value), usegmt=True)


def _aware_datetime(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.combine(value, time.min)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _unit_url(metadata: Mapping[str, Any]) -> str:
    for key in _URL_KEYS:
        if key in metadata:
            value = _clean_text(_scalar_text(metadata.get(key)))
            if value:
                return value
    return ""


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _scalar_text(value.model_dump())
    if isinstance(value, Mapping):
        return "; ".join(
            f"{key}: {_scalar_text(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if _scalar_text(item)
        )
    return str(value)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())
