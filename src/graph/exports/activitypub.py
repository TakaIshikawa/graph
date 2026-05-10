"""Export knowledge units to ActivityPub-compatible JSON format."""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from datetime import datetime

from graph.types.models import KnowledgeUnit


def export_units_to_activitypub(
    units: Iterable[KnowledgeUnit],
    *,
    context_url: str = "https://www.w3.org/ns/activitystreams",
) -> str:
    """Export units as an ActivityPub OrderedCollection of Note objects.

    Args:
        units: Knowledge units to export.
        context_url: The JSON-LD @context URL.

    Returns:
        JSON string of an ActivityPub OrderedCollection.
    """
    items: list[dict] = []
    for unit in units:
        note = _unit_to_note(unit)
        items.append(note)

    collection: dict = {
        "@context": context_url,
        "type": "OrderedCollection",
        "totalItems": len(items),
        "orderedItems": items,
    }
    return json.dumps(collection, ensure_ascii=False, indent=2)


def _unit_to_note(unit: KnowledgeUnit) -> dict:
    """Convert a KnowledgeUnit to an ActivityPub Note object."""
    note: dict = {
        "type": "Note",
        "name": unit.title,
        "content": html.escape(unit.content),
    }

    # Tags as Hashtag objects
    if unit.tags:
        note["tag"] = [
            {"type": "Hashtag", "name": f"#{tag}"}
            for tag in unit.tags
        ]

    # Published date from metadata or created_at
    published = _extract_published(unit)
    if published:
        note["published"] = published

    # Source with plaintext content
    note["source"] = {
        "content": unit.content,
        "mediaType": "text/plain",
    }

    return note


def _extract_published(unit: KnowledgeUnit) -> str | None:
    """Extract an ISO date string for the published field."""
    # Check metadata for date fields
    for key in ("date", "published", "created_at", "due_date"):
        val = unit.metadata.get(key)
        if val:
            if isinstance(val, datetime):
                return val.isoformat()
            return str(val)

    # Fall back to created_at
    if unit.created_at:
        return unit.created_at.isoformat()

    return None
