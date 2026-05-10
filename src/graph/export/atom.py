"""Atom (RFC 4287) feed export for knowledge units."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from graph.types.models import KnowledgeUnit

ATOM_NS = "http://www.w3.org/2005/Atom"


def export_units_to_atom(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    feed_title: str = "Knowledge Graph",
    feed_id: str = "urn:graph:feed",
) -> str:
    """Export units as an Atom XML feed.

    Args:
        units: Single unit or iterable of units.
        path: Optional file path to write the XML.
        feed_title: Title element for the feed.
        feed_id: ID element for the feed.

    Returns:
        Atom XML string.
    """
    if isinstance(units, KnowledgeUnit):
        unit_list = [units]
    else:
        unit_list = list(units)

    feed = Element("feed", xmlns=ATOM_NS)
    SubElement(feed, "title").text = feed_title
    SubElement(feed, "id").text = feed_id

    # Updated: latest unit updated_at, or now
    latest = _latest_updated(unit_list)
    SubElement(feed, "updated").text = latest.isoformat()

    for unit in unit_list:
        entry = SubElement(feed, "entry")
        SubElement(entry, "title").text = unit.title
        SubElement(entry, "id").text = unit.source_id or unit.id or ""

        content_el = SubElement(entry, "content")
        content_el.set("type", "html")
        content_el.text = unit.content

        updated = unit.updated_at or unit.created_at
        if updated:
            SubElement(entry, "updated").text = updated.isoformat()

        for tag in unit.tags:
            cat = SubElement(entry, "category")
            cat.set("term", tag)

    xml_bytes = tostring(feed, encoding="unicode", xml_declaration=True)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_bytes, encoding="utf-8")

    return xml_bytes


def _latest_updated(units: list[KnowledgeUnit]) -> datetime:
    """Find the latest updated_at among units, falling back to now."""
    dates = [u.updated_at for u in units if u.updated_at]
    if dates:
        return max(dates)
    return datetime.now(timezone.utc)
