"""Export adapter registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from graph.types.models import KnowledgeUnit

# Type alias for export functions
ExportFunction = Callable[[Iterable[KnowledgeUnit]], str]

_EXPORTERS: dict[str, ExportFunction] = {}


def register_exporter(name: str, func: ExportFunction) -> None:
    """
    Register an export function in the global registry.

    Args:
        name: Unique identifier for the exporter (e.g., "html", "json")
        func: Export function that takes units and returns a string
    """
    _EXPORTERS[name] = func


def get_exporter(name: str) -> ExportFunction:
    """
    Get an export function by name.

    Args:
        name: Exporter identifier

    Returns:
        Export function

    Raises:
        KeyError: If exporter name is not registered
    """
    if name not in _EXPORTERS:
        available = sorted(_EXPORTERS.keys())
        raise KeyError(f"Unknown exporter: {name}. Available: {available}")
    return _EXPORTERS[name]


def list_exporters() -> list[str]:
    """
    List all registered exporter names.

    Returns:
        Sorted list of exporter names
    """
    return sorted(_EXPORTERS.keys())


def export_units(
    name: str,
    units: Iterable[KnowledgeUnit],
    **kwargs: Any,
) -> str:
    """
    Export units using a registered exporter.

    Args:
        name: Exporter identifier
        units: Knowledge units to export
        **kwargs: Additional arguments passed to the export function

    Returns:
        Exported string representation
    """
    func = get_exporter(name)
    # Always pass kwargs - the function signature will handle them
    return func(units, **kwargs)


# Auto-register built-in exporters
from graph.exports.html import export_units_to_html
from graph.exports.ical import export_units_to_ical
from graph.exports.json_units import export_units_to_json
from graph.exports.markdown import export_units_to_markdown

register_exporter("html", export_units_to_html)
register_exporter("ical", export_units_to_ical)
register_exporter("json", export_units_to_json)
register_exporter("markdown", export_units_to_markdown)
