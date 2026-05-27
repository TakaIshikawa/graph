from __future__ import annotations

import json

from graph.adapters.hypothesis_annotations_json import HypothesisAnnotationsJsonAdapter
from graph.adapters.registry import get_adapter


def test_hypothesis_annotations_json_ingests_rows(tmp_path):
    export = tmp_path / "hypothesis.json"
    export.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "a1",
                        "text": "My note",
                        "quote": "Quoted text",
                        "uri": "https://example.com",
                        "document": {"title": ["Example"]},
                        "tags": ["Research"],
                        "created": "2024-01-01T00:00:00Z",
                        "updated": "2024-01-02T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = HypothesisAnnotationsJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Example"
    assert unit.metadata["quote"] == "Quoted text"
    assert unit.metadata["uri"] == "https://example.com"
    assert unit.tags == ["research"]
    assert get_adapter("hypothesis_annotations_json", path=str(export)).name == "hypothesis_annotations_json"


def test_hypothesis_annotations_json_handles_malformed_json(tmp_path):
    export = tmp_path / "bad.json"
    export.write_text("{", encoding="utf-8")

    assert HypothesisAnnotationsJsonAdapter(path=str(export)).ingest().units == []
