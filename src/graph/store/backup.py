"""Portable JSON backup export for graph stores."""

from __future__ import annotations

import base64
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from graph.store.migrations import SCHEMA_VERSION
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

BACKUP_FORMAT = "graph.store.backup.v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _unit_payload(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "id": unit.id,
        "source_project": _json_ready(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content": unit.content,
        "content_type": _json_ready(unit.content_type),
        "metadata": _json_ready(unit.metadata),
        "tags": _json_ready(unit.tags),
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": _json_ready(unit.created_at),
        "ingested_at": _json_ready(unit.ingested_at),
        "updated_at": _json_ready(unit.updated_at),
    }


def _edge_payload(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "from_unit_id": edge.from_unit_id,
        "to_unit_id": edge.to_unit_id,
        "relation": _json_ready(edge.relation),
        "weight": edge.weight,
        "source": _json_ready(edge.source),
        "metadata": _json_ready(edge.metadata),
        "created_at": _json_ready(edge.created_at),
    }


def _sync_state_payload(state: SyncState) -> dict[str, Any]:
    return {
        "source_project": state.source_project,
        "source_entity_type": state.source_entity_type,
        "last_sync_at": _json_ready(state.last_sync_at),
        "last_source_id": state.last_source_id,
        "items_synced": state.items_synced,
    }


def export_store_backup(
    store,
    path: str | Path | None = None,
    *,
    include_embeddings: bool = False,
) -> str:
    """Export all persisted graph store data as deterministic JSON text."""
    units = sorted(
        (_unit_payload(unit) for unit in store.get_all_units(limit=1_000_000_000)),
        key=lambda item: item["id"],
    )
    edges = sorted(
        (_edge_payload(edge) for edge in store.get_all_edges()),
        key=lambda item: item["id"],
    )
    sync_state = sorted(
        (_sync_state_payload(state) for state in store.get_all_sync_state()),
        key=lambda item: (item["source_project"], item["source_entity_type"]),
    )

    if include_embeddings:
        embeddings = {
            unit.id: base64.b64encode(embedding).decode("ascii")
            for unit, embedding in store.get_units_with_embeddings()
        }
        for unit in units:
            if unit["id"] in embeddings:
                unit["embedding"] = embeddings[unit["id"]]

    payload = {
        "metadata": {
            "format": BACKUP_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "include_embeddings": include_embeddings,
            "unit_count": len(units),
            "edge_count": len(edges),
            "sync_state_count": len(sync_state),
            "embedding_encoding": "base64" if include_embeddings else None,
        },
        "units": units,
        "edges": edges,
        "sync_state": sync_state,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    return text
