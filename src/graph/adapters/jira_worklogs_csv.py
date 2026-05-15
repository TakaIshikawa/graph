"""Adapter for Jira worklog CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class JiraWorklogsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "jira_worklogs_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["worklog"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "worklog" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        issue_key = first(row, "Issue key", "Issue Key", "Key", "issue_key")
        summary = first(row, "Issue summary", "Summary", "Issue Summary", "issue_summary")
        author = first(row, "Author", "Worklog author", "User", "Name")
        started_text = first(row, "Started", "Start date", "Start time", "Started at", "Date started", "Date")
        started = parse_datetime(started_text)
        time_spent = first(row, "Time spent", "Time Spent", "Time spent (seconds)", "Seconds", "Duration")
        seconds = self._seconds(row, time_spent)
        comment = first(row, "Comment", "Worklog comment", "Description", "Notes")
        project = first(row, "Project", "Project key", "Project name")
        url = first(row, "URL", "Issue URL", "Worklog URL", "Link")
        worklog_id = first(row, "Worklog ID", "Worklog id", "ID", "Id")
        if not any([issue_key, summary, author, started_text, time_spent, comment, worklog_id]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "worklog_id": worklog_id,
                "issue_key": issue_key,
                "issue_summary": summary,
                "author": author,
                "started_at": started.isoformat() if started else started_text,
                "time_spent": time_spent,
                "time_spent_seconds": seconds,
                "comment": comment,
                "project": project,
                "url": url,
                "source_url": url,
                "source_file": source_file,
            }
        )
        title = self._title(issue_key, summary, author, started)
        return KnowledgeUnit(
            source_project=SourceProject.JIRA_WORKLOGS_CSV,
            source_id=self._source_id(worklog_id, issue_key, author, started_text, time_spent, comment, index),
            source_entity_type="worklog",
            title=title,
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["jira", "worklog", project, issue_key] if tag)),
            created_at=started or now,
            updated_at=started or now,
        )

    def _seconds(self, row: dict[str, Any], time_spent: str) -> int | None:
        explicit = first(row, "Time spent (seconds)", "Time Spent Seconds", "Seconds")
        parsed = parse_duration_seconds(explicit)
        if parsed is not None:
            return parsed
        parsed = self._jira_duration_seconds(time_spent)
        if parsed is not None:
            return parsed
        return parse_duration_seconds(time_spent)

    def _jira_duration_seconds(self, value: str) -> int | None:
        text = value.strip().casefold()
        if not text:
            return None
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(w|week|weeks|d|day|days|h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)\b", text)
        if not matches:
            return None
        total = 0.0
        for raw, unit in matches:
            number = float(raw)
            if unit.startswith("w"):
                total += number * 5 * 8 * 3600
            elif unit.startswith("d"):
                total += number * 8 * 3600
            elif unit.startswith("h"):
                total += number * 3600
            elif unit.startswith("m"):
                total += number * 60
            else:
                total += number
        return int(round(total))

    def _source_id(self, worklog_id: str, issue_key: str, author: str, started: str, time_spent: str, comment: str, index: int) -> str:
        if worklog_id:
            return f"jira_worklogs_csv:{worklog_id}"
        return digest_source_id("jira_worklogs_csv", issue_key, author, started, time_spent, comment, index)

    def _title(self, issue_key: str, summary: str, author: str, started: datetime | None) -> str:
        subject = " ".join(part for part in [issue_key, summary] if part).strip() or "Jira worklog"
        suffix = " ".join(part for part in [author, started.date().isoformat() if started else ""] if part).strip()
        return f"{subject} ({suffix})" if suffix else subject

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Issue: {metadata.get('issue_key')}" if metadata.get("issue_key") else "",
            f"Summary: {metadata.get('issue_summary')}" if metadata.get("issue_summary") else "",
            f"Author: {metadata.get('author')}" if metadata.get("author") else "",
            f"Started: {metadata.get('started_at')}" if metadata.get("started_at") else "",
            f"Time spent: {metadata.get('time_spent')}" if metadata.get("time_spent") else "",
            f"Comment: {metadata.get('comment')}" if metadata.get("comment") else "",
            f"URL: {metadata.get('url')}" if metadata.get("url") else "",
        ]
        return "\n".join(part for part in parts if part)
