from __future__ import annotations

import json

from graph.adapters.github_discussions_json import GithubDiscussionsJsonAdapter
from graph.adapters.registry import get_adapter


def test_github_discussions_json_flattens_graphql_and_answer_metadata(tmp_path):
    path = tmp_path / "discussions.json"
    path.write_text(
        json.dumps(
            {
                "data": {
                    "repository": {
                        "discussions": {
                            "edges": [
                                {
                                    "node": {
                                        "id": "D1",
                                        "number": 7,
                                        "title": "Import data",
                                        "bodyText": "Discussion body",
                                        "url": "https://github.test/d/7",
                                        "category": {"name": "Q&A"},
                                        "author": {"login": "ada"},
                                        "createdAt": "2026-01-01T00:00:00Z",
                                        "updatedAt": "2026-01-02T00:00:00Z",
                                        "isAnswered": True,
                                        "answer": {"id": "A1", "url": "https://github.test/a/1"},
                                        "labels": {"nodes": [{"name": "bug"}]},
                                        "comments": {"totalCount": 3},
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    unit = GithubDiscussionsJsonAdapter(str(path)).ingest().units[0]

    assert unit.title == "Import data"
    assert "Discussion body" in unit.content
    assert unit.metadata["accepted_answer_id"] == "A1"
    assert unit.metadata["accepted_answer_url"] == "https://github.test/a/1"
    assert unit.metadata["labels"] == ["bug"]
    assert unit.metadata["category"] == "Q&A"
    assert unit.metadata["comment_count"] == 3
    assert isinstance(get_adapter("github_discussions_json"), GithubDiscussionsJsonAdapter)
