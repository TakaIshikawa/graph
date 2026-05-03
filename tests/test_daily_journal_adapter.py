from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from graph.adapters.daily_journal import DailyJournalAdapter
from graph.types.enums import ContentType, SourceProject


def test_ingests_markdown_journal_with_frontmatter_date_and_tags(tmp_path):
    journal = tmp_path / "notes" / "2025-01-02.md"
    journal.parent.mkdir()
    journal.write_text(
        """---
date: 2025-01-03
title: 2025-01-03 Friday
tags:
  - Reflection
  - "#Work"
mood: focused
---
# Morning

Shipped the parser.
""",
        encoding="utf-8",
    )

    result = DailyJournalAdapter(path=str(tmp_path), source_id_root=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    expected_digest = hashlib.sha256(
        "notes/2025-01-02.md|2025-01-03".encode("utf-8")
    ).hexdigest()[:16]
    assert unit.source_project == SourceProject.DAILY_JOURNAL
    assert unit.source_entity_type == "journal_entry"
    assert unit.source_id == f"daily_journal:2025-01-03:{expected_digest}"
    assert unit.title == "2025-01-03 Friday"
    assert unit.content == "# Morning\n\nShipped the parser.\n"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["journal", "reflection", "work"]
    assert unit.created_at == datetime(2025, 1, 3, tzinfo=timezone.utc)
    assert unit.metadata == {
        "source_file": "notes/2025-01-02.md",
        "journal_date": "2025-01-03",
        "frontmatter": {
            "date": "2025-01-03",
            "title": "2025-01-03 Friday",
            "tags": ["Reflection", "#Work"],
            "mood": "focused",
        },
    }


def test_frontmatter_title_date_takes_precedence_over_filename(tmp_path):
    journal = tmp_path / "2025-04-01.md"
    journal.write_text(
        """---
title: Journal for 2025-04-02
tags: personal, journal
---
Plain body.
""",
        encoding="utf-8",
    )

    result = DailyJournalAdapter(path=str(journal)).ingest()

    unit = result.units[0]
    assert unit.metadata["journal_date"] == "2025-04-02"
    assert unit.tags == ["journal", "personal"]
    assert unit.created_at == datetime(2025, 4, 2, tzinfo=timezone.utc)


def test_filename_date_is_used_without_frontmatter(tmp_path):
    journal = tmp_path / "daily-2025-05-06.markdown"
    journal.write_text("Plain Markdown body.\n", encoding="utf-8")

    result = DailyJournalAdapter(root_path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert unit.title == "daily-2025-05-06"
    assert unit.content == "Plain Markdown body.\n"
    assert unit.metadata["journal_date"] == "2025-05-06"
    assert unit.metadata["frontmatter"] == {}
    assert unit.tags == ["journal"]


def test_missing_date_raises_value_error(tmp_path):
    journal = tmp_path / "untitled.md"
    journal.write_text("# No date here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing journal date"):
        DailyJournalAdapter(path=str(journal)).ingest()


def test_malformed_frontmatter_date_raises_value_error(tmp_path):
    journal = tmp_path / "2025-06-01.md"
    journal.write_text(
        """---
date: not-a-date
---
Body.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid frontmatter date"):
        DailyJournalAdapter(path=str(journal)).ingest()


def test_entity_type_filter_returns_empty_result(tmp_path):
    journal = tmp_path / "2025-07-01.md"
    journal.write_text("Body.\n", encoding="utf-8")

    result = DailyJournalAdapter(path=str(tmp_path)).ingest(entity_types=["note"])

    assert result.units == []
    assert result.edges == []
