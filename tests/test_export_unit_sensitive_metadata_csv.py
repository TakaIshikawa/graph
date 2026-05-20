from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_sensitive_metadata_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: str = "Project") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_sensitive_metadata_csv_empty_input_has_header_only():
    assert export_unit_sensitive_metadata_csv([]) == (
        "source,unit_id,title,metadata_key,redacted_value,risk_label\n"
    )


def test_unit_sensitive_metadata_csv_redacts_default_sensitive_keys_deterministically():
    text = export_unit_sensitive_metadata_csv(
        [
            unit("b", metadata={"safe": "visible", "apiKey": "abcd"}),
            unit("a", metadata={"password": "", "private-key": "very-long-secret"}),
        ]
    )

    assert rows(text) == [
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "metadata_key": "password",
            "redacted_value": "empty",
            "risk_label": "high",
        },
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "metadata_key": "private-key",
            "redacted_value": "long:16",
            "risk_label": "high",
        },
        {
            "source": "Project",
            "unit_id": "b",
            "title": "Title b",
            "metadata_key": "apiKey",
            "redacted_value": "short:4",
            "risk_label": "medium",
        },
    ]


def test_unit_sensitive_metadata_csv_supports_custom_patterns_and_mappings():
    text = export_unit_sensitive_metadata_csv(
        [{"id": "m1", "source_project": "Map", "title": "Mapped", "metadata": {"session": "abcdef", "token": "ignored"}}],
        key_patterns=["session"],
    )

    assert rows(text) == [
        {
            "source": "Map",
            "unit_id": "m1",
            "title": "Mapped",
            "metadata_key": "session",
            "redacted_value": "long:6",
            "risk_label": "review",
        }
    ]


@pytest.mark.parametrize("patterns", [[], [""], ["  "]])
def test_unit_sensitive_metadata_csv_validates_non_empty_patterns(patterns):
    with pytest.raises(ValueError, match="key_patterns"):
        export_unit_sensitive_metadata_csv([], key_patterns=patterns)


def test_unit_sensitive_metadata_csv_path_mode_writes_parent_dirs_and_stats(tmp_path):
    path = tmp_path / "nested" / "sensitive.csv"
    units = [unit("a", metadata={"token": "abc"})]

    expected = export_unit_sensitive_metadata_csv(units)
    stats = export_unit_sensitive_metadata_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
