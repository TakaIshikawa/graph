"""Extract sample size signals from evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, value

_PATTERNS = (
    re.compile(r"\b[Nn]\s*=\s*([0-9][0-9,]*)\b"),
    re.compile(r"\b([0-9][0-9,]*)\s+(participants|respondents|records|observations)\b", re.I),
)


def extract_evidence_sample_size_signals(evidence: Iterable[Any]) -> dict[str, Any]:
    records = []
    sizes = []
    for index, item in enumerate(evidence):
        signals = []
        meta_size = _int(value(item, "sample_size"))
        if meta_size is not None:
            signals.append({"source": "metadata", "value": meta_size})
        for pattern in _PATTERNS:
            for match in pattern.finditer(content_text(item)):
                signals.append({"source": "text", "value": _int(match.group(1)), "cue": match.group(0)})
        values = [signal["value"] for signal in signals if signal["value"] is not None]
        sizes.extend(values)
        records.append({"id": result_id(item, index), "sample_sizes": values, "signals": signals})
    return {
        "records": records,
        "summary": {
            "total_records": len(records),
            "records_with_sample_size": sum(1 for r in records if r["sample_sizes"]),
            "unknown_count": sum(1 for r in records if not r["sample_sizes"]),
            "min": min(sizes) if sizes else None,
            "max": max(sizes) if sizes else None,
        },
    }


def _int(value_: Any) -> int | None:
    try:
        return int(str(value_).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
