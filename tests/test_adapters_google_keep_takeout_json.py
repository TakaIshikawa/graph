import json

from graph.adapters import GoogleKeepTakeoutJsonAdapter


def test_google_keep_takeout_json_ingests_text_checklists_labels_and_timestamps(tmp_path):
    path = tmp_path / "note.json"
    path.write_text(json.dumps({"title": "", "textContent": "Body", "listContent": [{"text": "Done", "isChecked": True}, {"text": "Todo", "isChecked": False}], "labels": [{"name": "work"}], "color": "yellow", "isPinned": True, "isArchived": False, "createdTimestampUsec": 1735689600000000, "userEditedTimestampUsec": 1735776000000000, "attachments": [{"filePath": "a.png"}]}), encoding="utf-8")

    unit = GoogleKeepTakeoutJsonAdapter(str(path)).ingest().units[0]
    again = GoogleKeepTakeoutJsonAdapter(str(path)).ingest().units[0]

    assert unit.title == "Body"
    assert unit.content == "Body\n[x] Done\n[ ] Todo\nLabels: work"
    assert unit.metadata["checklist"] == [{"text": "Done", "checked": True}, {"text": "Todo", "checked": False}]
    assert unit.metadata["isPinned"] is True
    assert unit.tags == ["work"]
    assert unit.source_id == again.source_id
