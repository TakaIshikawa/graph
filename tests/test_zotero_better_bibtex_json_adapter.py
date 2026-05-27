from __future__ import annotations

import json

from graph.adapters.registry import get_adapter
from graph.adapters.zotero_better_bibtex_json import ZoteroBetterBibtexJsonAdapter


def test_zotero_better_bibtex_json_ingests_arrays_and_bibliography_metadata(tmp_path):
    path = tmp_path / "zotero.json"
    path.write_text(
        json.dumps(
            [
                {
                    "citationKey": "doe2024",
                    "itemType": "journalArticle",
                    "title": "Graph Notes",
                    "abstractNote": "An abstract.",
                    "creators": [{"lastName": "Doe"}],
                    "DOI": "10.1/example",
                    "url": "https://example.test",
                    "date": "2024",
                    "tags": [{"tag": "notes"}],
                    "collections": ["Inbox"],
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = ZoteroBetterBibtexJsonAdapter(str(path)).ingest().units[0]

    assert unit.source_project == "zotero_better_bibtex_json"
    assert unit.source_id.startswith("zotero_better_bibtex_json:")
    assert unit.title == "Graph Notes"
    assert unit.content == "An abstract."
    assert unit.metadata["citation_key"] == "doe2024"
    assert unit.metadata["item_type"] == "journalArticle"
    assert unit.metadata["doi"] == "10.1/example"
    assert unit.metadata["tags"] == ["notes"]
    assert unit.metadata["collections"] == ["Inbox"]
    assert isinstance(get_adapter("zotero_better_bibtex_json"), ZoteroBetterBibtexJsonAdapter)


def test_zotero_better_bibtex_json_accepts_item_container_and_fallback_content(tmp_path):
    path = tmp_path / "zotero.json"
    path.write_text(json.dumps({"items": [{"key": "abc", "title": "No Abstract", "creators": ["Ada"], "date": "2026"}]}), encoding="utf-8")

    unit = ZoteroBetterBibtexJsonAdapter(str(path)).ingest().units[0]

    assert "No Abstract" in unit.content
    assert "Ada" in unit.content
