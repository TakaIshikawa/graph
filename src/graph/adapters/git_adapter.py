"""Adapter for local Git repository commit logs."""

from __future__ import annotations

import re
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


FIELD_SEPARATOR = "\x1f"
RECORD_SEPARATOR = "\x1e"


class GitAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "git"

    @property
    def entity_types(self) -> list[str]:
        return ["commit"]

    def __init__(self, repos: str = "") -> None:
        self.repos = repos

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "commit" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        skipped_repos = 0
        for repo in self._discover_repos():
            repo_root = self._repo_root(repo)
            if repo_root is None:
                skipped_repos += 1
                continue
            for record in self._commit_records(repo_root):
                unit = self._unit_from_record(repo_root, record)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                result.units.append(unit)

        if skipped_repos:
            suffix = "y" if skipped_repos == 1 else "ies"
            warnings.warn(
                f"Skipped {skipped_repos} invalid Git repositor{suffix}.",
                stacklevel=2,
            )

        return result

    def _discover_repos(self) -> list[Path]:
        return [
            Path(source.strip()).expanduser()
            for source in re.split(r"[\n,]", self.repos)
            if source.strip()
        ]

    def _repo_root(self, repo: Path) -> Path | None:
        if not repo.exists():
            return None
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        root = completed.stdout.strip()
        return Path(root).resolve() if root else None

    def _commit_records(self, repo: Path) -> list[dict[str, str]]:
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "log",
                    "--date=iso-strict",
                    f"--pretty=format:%H{FIELD_SEPARATOR}%an{FIELD_SEPARATOR}%ae"
                    f"{FIELD_SEPARATOR}%aI{FIELD_SEPARATOR}%cI{FIELD_SEPARATOR}%D"
                    f"{FIELD_SEPARATOR}%B{RECORD_SEPARATOR}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return []

        records: list[dict[str, str]] = []
        for raw_record in completed.stdout.split(RECORD_SEPARATOR):
            raw_record = raw_record.strip()
            if not raw_record:
                continue
            fields = raw_record.split(FIELD_SEPARATOR, 6)
            if len(fields) != 7:
                continue
            sha, author, email, author_date, commit_date, refs, message = fields
            records.append(
                {
                    "sha": sha.strip(),
                    "author": author.strip(),
                    "email": email.strip(),
                    "author_date": author_date.strip(),
                    "commit_date": commit_date.strip(),
                    "refs": refs.strip(),
                    "message": message.strip(),
                }
            )
        return records

    def _unit_from_record(self, repo: Path, record: dict[str, str]) -> KnowledgeUnit | None:
        sha = record["sha"]
        if not sha:
            return None

        repo_name = repo.name
        message = record["message"] or sha
        title = message.splitlines()[0].strip() or sha
        author_date = self._parse_datetime(record["author_date"])
        commit_date = self._parse_datetime(record["commit_date"])
        created_at = author_date or commit_date or datetime.now(timezone.utc)
        refs = [ref.strip() for ref in record["refs"].split(",") if ref.strip()]

        metadata = {
            "sha": sha,
            "author": record["author"],
            "email": record["email"],
            "repo_path": str(repo),
            "repo_name": repo_name,
            "commit_date": commit_date.isoformat() if commit_date else None,
        }
        if refs:
            metadata["refs"] = refs

        unit = KnowledgeUnit(
            source_project=SourceProject.GIT,
            source_id=f"{repo_name}:{sha}",
            source_entity_type="commit",
            title=title,
            content=message,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["git", repo_name],
            created_at=created_at,
        )
        if commit_date is not None:
            unit.updated_at = commit_date
        return unit

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
