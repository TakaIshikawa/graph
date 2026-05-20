from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.stackoverflow_answers_json import StackOverflowAnswersJsonAdapter
from graph.types.models import SyncState


def test_stackoverflow_answers_json_ingests_top_level_array(tmp_path):
    export = tmp_path / "answers.json"
    export.write_text(
        json.dumps(
            [
                {
                    "answer_id": 123,
                    "question_id": 456,
                    "question_title": "How do I parse CSV in Python?",
                    "body": "Use the csv module.",
                    "score": 42,
                    "is_accepted": True,
                    "link": "https://stackoverflow.com/a/123",
                    "tags": ["python", "csv"],
                    "owner": {"display_name": "Ada", "user_id": 99, "link": "https://stackoverflow.com/users/99/ada"},
                    "creation_date": 1735689600,
                    "last_edit_date": "2025-01-02T03:04:05Z",
                    "license": "CC BY-SA 4.0",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = StackOverflowAnswersJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "stackoverflow_answers_json"
    assert unit.source_id == "stackoverflow_answers_json:answer:123"
    assert unit.source_entity_type == "answer"
    assert unit.title == "How do I parse CSV in Python?"
    assert unit.metadata["answer_id"] == 123
    assert unit.metadata["question_id"] == 456
    assert unit.metadata["question_title"] == "How do I parse CSV in Python?"
    assert unit.metadata["score"] == 42
    assert unit.metadata["is_accepted"] is True
    assert unit.metadata["url"] == "https://stackoverflow.com/a/123"
    assert unit.metadata["tags"] == ["python", "csv"]
    assert unit.metadata["owner"]["display_name"] == "Ada"
    assert unit.metadata["created_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["updated_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["license"] == "CC BY-SA 4.0"
    assert unit.metadata["source_file"] == "answers.json"
    assert unit.metadata["raw"]["body"] == "Use the csv module."
    assert "Use the csv module." in unit.content


def test_stackoverflow_answers_json_containers_filters_dedupe_and_registry(tmp_path):
    good = tmp_path / "answers.json"
    bad = tmp_path / "bad.json"
    good.write_text(
        json.dumps(
            {
                "data": {
                    "items": [
                        {"answerId": "1", "questionTitle": "Old", "bodyMarkdown": "old", "updatedAt": "2025-01-01"},
                        {"answerId": "2", "questionId": "20", "questionTitle": "New", "bodyMarkdown": "new", "tags": "python|json", "accepted": "yes", "updatedAt": "2025-01-03"},
                        {"answerId": "2", "questionId": "20", "questionTitle": "Newer", "bodyMarkdown": "newer", "updatedAt": "2025-01-04"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    bad.write_text("{not json", encoding="utf-8")
    since = SyncState(source_project="stackoverflow_answers_json", source_entity_type="answer", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))

    result = StackOverflowAnswersJsonAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = StackOverflowAnswersJsonAdapter(path=str(tmp_path)).ingest(entity_types=["question"])

    assert [unit.source_id for unit in result.units] == ["stackoverflow_answers_json:answer:2"]
    assert result.units[0].title == "Newer"
    assert skipped.units == []
    assert get_adapter("stackoverflow-answers-json", path=str(tmp_path)).name == "stackoverflow_answers_json"


def test_stackoverflow_answers_json_fallback_id_is_deterministic(tmp_path):
    export = tmp_path / "answers.json"
    export.write_text(json.dumps({"answers": [{"question_id": 456, "body": "No answer id", "url": "https://stackoverflow.com/a/no-id"}]}), encoding="utf-8-sig")

    first = StackOverflowAnswersJsonAdapter(path=str(export)).ingest().units[0]
    second = StackOverflowAnswersJsonAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
