from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from graph.adapters.hypothesis import HypothesisAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_hypothesis_ingests_annotations_from_rows_export(tmp_path):
    export = tmp_path / "hypothesis.json"
    export.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "ann-2",
                        "uri": "https://example.com/two",
                        "user": "acct:taka@hypothes.is",
                        "group": "abc123",
                        "text": "Second note",
                        "tags": ["PKM", "#Reading"],
                        "created": "2025-01-02T10:00:00Z",
                        "updated": "2025-01-03T10:00:00Z",
                        "document": {"title": ["Second Page"]},
                        "target": [
                            {
                                "source": "https://example.com/two",
                                "selector": [
                                    {"type": "TextQuoteSelector", "exact": "Quoted text two"}
                                ],
                            }
                        ],
                    },
                    {
                        "id": "ann-1",
                        "uri": "https://example.com/one",
                        "user": "acct:taka@hypothes.is",
                        "group": "abc123",
                        "text": "First note",
                        "tags": ["Research"],
                        "created": "2025-01-01T10:00:00Z",
                        "updated": "2025-01-01T11:00:00Z",
                        "document": {"title": ["First Page"]},
                        "target": [
                            {
                                "selector": {
                                    "type": "TextQuoteSelector",
                                    "exact": "Quoted text one",
                                }
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = HypothesisAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "hypothesis:ann-1",
        "hypothesis:ann-2",
    ]
    unit = result.units[0]
    assert unit.source_project == SourceProject.HYPOTHESIS
    assert unit.source_entity_type == "annotation"
    assert unit.title == "First Page"
    assert unit.content_type == ContentType.INSIGHT
    assert "First note" in unit.content
    assert "Quote: Quoted text one" in unit.content
    assert "URL: https://example.com/one" in unit.content
    assert unit.metadata["uri"] == "https://example.com/one"
    assert unit.metadata["group"] == "abc123"
    assert unit.metadata["user"] == "acct:taka@hypothes.is"
    assert unit.metadata["created"] == "2025-01-01T10:00:00Z"
    assert unit.metadata["updated"] == "2025-01-01T11:00:00Z"
    assert unit.metadata["quote"] == "Quoted text one"
    assert unit.metadata["text"] == "First note"
    assert unit.metadata["tags"] == ["Research"]
    assert unit.tags == ["Research"]
    assert unit.created_at == datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 1, 11, tzinfo=timezone.utc)


def test_hypothesis_uses_stable_hash_source_id_when_id_is_missing(tmp_path):
    export = tmp_path / "hypothesis.json"
    annotation = {
        "uri": "https://example.com/no-id",
        "text": "A note",
        "created": "2025-01-01T00:00:00Z",
        "target": [{"selector": [{"exact": "A quote"}]}],
    }
    export.write_text(json.dumps([annotation]), encoding="utf-8")

    first = HypothesisAdapter(path=str(export)).ingest().units[0]
    second = HypothesisAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("hypothesis:")
    assert first.title == "https://example.com/no-id"
    assert first.metadata["quote"] == "A quote"


def test_hypothesis_since_filters_updated_annotations(tmp_path):
    export = tmp_path / "hypothesis.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "old",
                    "uri": "https://example.com/old",
                    "updated": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "new",
                    "uri": "https://example.com/new",
                    "updated": "2025-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="hypothesis",
        source_entity_type="annotation",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = HypothesisAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["hypothesis:new"]


def test_hypothesis_empty_and_malformed_exports_return_empty(tmp_path):
    for payload in ([], {}, {"rows": ["not a dict", {"unrelated": "shape"}]}, "not an object"):
        export = tmp_path / f"{len(str(payload))}.json"
        export.write_text(json.dumps(payload), encoding="utf-8")

        result = HypothesisAdapter(path=str(export)).ingest()

        assert result.units == []
        assert result.edges == []


def test_hypothesis_invalid_json_raises(tmp_path):
    export = tmp_path / "bad.json"
    export.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        HypothesisAdapter(path=str(export)).ingest()


def test_hypothesis_respects_entity_types(tmp_path):
    export = tmp_path / "hypothesis.json"
    export.write_text(json.dumps([{"id": "ann", "uri": "https://example.com"}]), encoding="utf-8")

    result = HypothesisAdapter(path=str(export)).ingest(entity_types=["saved_item"])

    assert result.units == []
    assert result.edges == []


def test_hypothesis_adapter_is_registered():
    assert "hypothesis" in list_adapters()
    adapter = get_adapter("hypothesis", path="/tmp/hypothesis.json")
    assert adapter.name == "hypothesis"
