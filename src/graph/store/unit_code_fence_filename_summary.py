"""Summarize code fences that declare filenames."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id
from graph.export.unit_code_fence_filename_csv import _rows as _filename_rows


def summarize_unit_code_fence_filenames(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = total_fences = 0
    filenames: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        total_units += 1
        for row in _filename_rows(unit):
            total_fences += 1
            filename = str(row["filename"])
            filenames[filename] += 1
            extensions[_extension(filename)] += 1
            if len(samples) < sample_limit:
                samples.append({"unit_id": unit_id(unit), "line": row["line"], "filename": filename})
    duplicates = [{"filename": name, "count": filenames[name]} for name in sorted(filenames, key=sort_key) if filenames[name] > 1]
    return {"total_units": total_units, "filename_fence_count": total_fences, "extension_counts": _counter_rows(extensions, "extension"), "duplicate_filename_counts": duplicates, "samples": samples}


def _extension(filename: str) -> str:
    suffix = Path(filename).suffix.casefold()
    return suffix[1:] if suffix else ""


def _counter_rows(counter: Counter[str], key: str) -> list[dict[str, Any]]:
    return [{key: name, "count": counter[name]} for name in sorted(counter, key=sort_key)]
