"""Adapter for LinkedIn archive exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class LinkedInArchiveAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linkedin_archive"

    @property
    def entity_types(self) -> list[str]:
        return ["connection", "message", "position"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        root = Path(self.path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_datetime(since) if since else None
        ingest_connections = not entity_types or "connection" in entity_types
        ingest_messages = not entity_types or "message" in entity_types
        ingest_positions = not entity_types or "position" in entity_types

        # Track references for creating edges
        user_connections: list[str] = []
        message_conversations: dict[str, str] = {}
        user_positions: list[str] = []
        position_companies: dict[str, str] = {}

        if ingest_connections:
            for path in self._find_connections_files(root):
                for connection_data in self._read_csv(path):
                    unit = self._connection_unit(connection_data, path, root)
                    if unit is None:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)
                    user_connections.append(unit.source_id)

        if ingest_messages:
            for path in self._find_message_files(root):
                for message_data in self._read_csv(path):
                    unit = self._message_unit(message_data, path, root)
                    if unit is None:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)
                    conversation_id = self._first(message_data, "CONVERSATION ID", "conversation_id")
                    if conversation_id:
                        message_conversations[unit.source_id] = conversation_id

        if ingest_positions:
            for path in self._find_position_files(root):
                for position_data in self._read_csv(path):
                    unit = self._position_unit(position_data, path, root)
                    if unit is None:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)
                    user_positions.append(unit.source_id)
                    company = self._first(position_data, "Company Name", "company_name")
                    if company:
                        position_companies[unit.source_id] = company

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        included_source_ids = {unit.source_id for unit in result.units}

        # Create edges
        emitted_edges: set[tuple[str, str, str]] = set()

        # Connection edges (user -> connection)
        for connection_id in user_connections:
            if connection_id not in included_source_ids:
                continue
            edge_key = ("linkedin_user", connection_id, "linkedin_connection")
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id("linkedin_user", connection_id, "linkedin_connection"),
                    from_unit_id="linkedin_user:self",
                    to_unit_id=connection_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.LINKEDIN_ARCHIVE.value,
                        "from_entity_type": "user",
                        "to_entity_type": "connection",
                        "relation_type": "linkedin_connection",
                    },
                )
            )

        # Conversation edges (message -> message in same conversation)
        conversation_groups: dict[str, list[str]] = {}
        for source_id, conv_id in message_conversations.items():
            if source_id not in included_source_ids:
                continue
            conversation_groups.setdefault(conv_id, []).append(source_id)

        for conv_id, message_ids in conversation_groups.items():
            if len(message_ids) < 2:
                continue
            message_ids.sort()
            for i in range(len(message_ids) - 1):
                from_id = message_ids[i + 1]
                to_id = message_ids[i]
                edge_key = (from_id, to_id, "linkedin_conversation")
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(from_id, to_id, "linkedin_conversation"),
                        from_unit_id=from_id,
                        to_unit_id=to_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.LINKEDIN_ARCHIVE.value,
                            "from_entity_type": "message",
                            "to_entity_type": "message",
                            "relation_type": "linkedin_conversation",
                            "conversation_id": conv_id,
                        },
                    )
                )

        # Position edges (user -> position, position -> company)
        for position_id in user_positions:
            if position_id not in included_source_ids:
                continue
            edge_key = ("linkedin_user", position_id, "linkedin_position")
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id("linkedin_user", position_id, "linkedin_position"),
                    from_unit_id="linkedin_user:self",
                    to_unit_id=position_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.LINKEDIN_ARCHIVE.value,
                        "from_entity_type": "user",
                        "to_entity_type": "position",
                        "relation_type": "linkedin_position",
                    },
                )
            )

        # Company edges (position -> company)
        for position_id, company_name in position_companies.items():
            if position_id not in included_source_ids:
                continue
            edge_key = (position_id, company_name, "linkedin_company")
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id(position_id, company_name, "linkedin_company"),
                    from_unit_id=position_id,
                    to_unit_id=f"linkedin_company:{self._company_id(company_name)}",
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.LINKEDIN_ARCHIVE.value,
                        "from_entity_type": "position",
                        "to_entity_type": "company",
                        "relation_type": "linkedin_company",
                        "company_name": company_name,
                    },
                )
            )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _find_connections_files(self, root: Path) -> list[Path]:
        """Find connections CSV files."""
        return self._find_csv_files(root, ["Connections.csv", "connections.csv"])

    def _find_message_files(self, root: Path) -> list[Path]:
        """Find messages CSV files."""
        return self._find_csv_files(root, ["messages.csv", "Messages.csv"])

    def _find_position_files(self, root: Path) -> list[Path]:
        """Find positions CSV files."""
        return self._find_csv_files(
            root,
            ["Positions.csv", "positions.csv", "Profile.csv", "profile.csv"],
        )

    def _find_csv_files(self, root: Path, names: list[str]) -> list[Path]:
        order = {name.lower(): index for index, name in enumerate(names)}
        matches: dict[str, Path] = {}
        for child in root.iterdir():
            key = child.name.lower()
            if key in order and key not in matches and child.is_file():
                matches[key] = child
        return sorted(matches.values(), key=lambda path: order[path.name.lower()])

    def _read_csv(self, path: Path) -> list[dict[str, Any]]:
        """Read and parse CSV file."""
        rows: list[dict[str, Any]] = []
        try:
            # Try different encodings
            for encoding in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
                try:
                    with path.open("r", encoding=encoding, newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            rows.append(dict(row))
                    break
                except UnicodeDecodeError:
                    continue
        except (OSError, csv.Error):
            return []
        return rows

    def _connection_unit(
        self,
        connection: dict[str, Any],
        path: Path,
        root: Path,
    ) -> KnowledgeUnit | None:
        """Create KnowledgeUnit from LinkedIn connection."""
        first_name = self._first(connection, "First Name", "first_name")
        last_name = self._first(connection, "Last Name", "last_name")

        if not first_name and not last_name:
            return None

        email = self._first(connection, "Email Address", "email")
        company = self._first(connection, "Company", "company")
        position = self._first(connection, "Position", "position")
        connected_on = self._first(connection, "Connected On", "connected_on")

        # Parse date
        created_at = self._parse_date(connected_on) or datetime.now(timezone.utc)

        # Generate connection ID
        connection_id = self._first(connection, "id", "connection_id")
        if not connection_id:
            raw = f"{first_name}:{last_name}:{email}:{connected_on}"
            connection_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"linkedin_archive:connection:{connection_id}"
        source_path = self._relative_path(path, root)

        # Build metadata
        metadata: dict[str, Any] = {
            "connection_id": connection_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "company": company,
            "position": position,
            "connected_on": connected_on,
            "source_path": source_path,
        }

        # Create title
        full_name = f"{first_name} {last_name}".strip()
        title_parts = [full_name]
        if position and company:
            title_parts.append(f"{position} at {company}")
        elif position:
            title_parts.append(position)
        elif company:
            title_parts.append(company)
        display_title = " - ".join(title_parts) if title_parts else "LinkedIn Connection"

        # Build tags
        tags = ["linkedin", "linkedin-connection"]
        if company:
            tags.append(f"company-{self._tag_value(company)}")

        return KnowledgeUnit(
            source_project=SourceProject.LINKEDIN_ARCHIVE,
            source_id=source_id,
            source_entity_type="connection",
            title=display_title,
            content=self._connection_content(full_name, email, company, position, connected_on),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(tags),
            created_at=created_at,
            updated_at=created_at,
        )

    def _message_unit(
        self,
        message: dict[str, Any],
        path: Path,
        root: Path,
    ) -> KnowledgeUnit | None:
        """Create KnowledgeUnit from LinkedIn message."""
        content = self._first(message, "CONTENT", "content")
        from_name = self._first(message, "FROM", "from")
        to_name = self._first(message, "TO", "to")

        if not content:
            return None

        date_str = self._first(message, "DATE", "date")
        created_at = self._parse_date(date_str) or datetime.now(timezone.utc)

        subject = self._first(message, "SUBJECT", "subject")
        conversation_id = self._first(message, "CONVERSATION ID", "conversation_id")

        # Generate message ID
        message_id = self._first(message, "id", "message_id")
        if not message_id:
            raw = f"{conversation_id}:{from_name}:{to_name}:{content}:{date_str}"
            message_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"linkedin_archive:message:{message_id}"
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "from": from_name,
            "to": to_name,
            "date": date_str,
            "subject": subject,
            "source_path": source_path,
        }

        # Create title
        if subject:
            display_title = subject
        else:
            display_title = content
        if len(display_title) > 80:
            display_title = f"{display_title[:77].rstrip()}..."

        return KnowledgeUnit(
            source_project=SourceProject.LINKEDIN_ARCHIVE,
            source_id=source_id,
            source_entity_type="message",
            title=display_title,
            content=self._message_content(from_name, to_name, subject, content, date_str),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(["linkedin", "linkedin-message"]),
            created_at=created_at,
            updated_at=created_at,
        )

    def _position_unit(
        self,
        position: dict[str, Any],
        path: Path,
        root: Path,
    ) -> KnowledgeUnit | None:
        """Create KnowledgeUnit from LinkedIn position."""
        company_name = self._first(position, "Company Name", "company_name")
        title = self._first(position, "Title", "title")

        if not company_name and not title:
            return None

        description = self._first(position, "Description", "description")
        location = self._first(position, "Location", "location")
        started_on = self._first(position, "Started On", "started_on")
        finished_on = self._first(position, "Finished On", "finished_on")

        # Parse dates
        start_date = self._parse_date(started_on)
        created_at = start_date or datetime.now(timezone.utc)

        # Generate position ID
        position_id = self._first(position, "id", "position_id")
        if not position_id:
            raw = f"{company_name}:{title}:{started_on}:{finished_on}"
            position_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"linkedin_archive:position:{position_id}"
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "position_id": position_id,
            "company_name": company_name,
            "title": title,
            "description": description,
            "location": location,
            "started_on": started_on,
            "finished_on": finished_on,
            "source_path": source_path,
        }

        # Create display title
        title_parts = []
        if title:
            title_parts.append(title)
        if company_name:
            title_parts.append(f"at {company_name}")
        display_title = " ".join(title_parts) if title_parts else "LinkedIn Position"

        # Build tags
        tags = ["linkedin", "linkedin-position"]
        if company_name:
            tags.append(f"company-{self._tag_value(company_name)}")

        return KnowledgeUnit(
            source_project=SourceProject.LINKEDIN_ARCHIVE,
            source_id=source_id,
            source_entity_type="position",
            title=display_title,
            content=self._position_content(title, company_name, description, location, started_on, finished_on),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(tags),
            created_at=created_at,
            updated_at=created_at,
        )

    def _connection_content(
        self,
        full_name: str,
        email: str,
        company: str,
        position: str,
        connected_on: str,
    ) -> str:
        """Build content string for connection."""
        parts = []
        if full_name:
            parts.append(f"Name: {full_name}")
        if email:
            parts.append(f"Email: {email}")
        if position:
            parts.append(f"Position: {position}")
        if company:
            parts.append(f"Company: {company}")
        if connected_on:
            parts.append(f"Connected: {connected_on}")
        return "\n".join(parts)

    def _message_content(
        self,
        from_name: str,
        to_name: str,
        subject: str,
        content: str,
        date: str,
    ) -> str:
        """Build content string for message."""
        parts = []
        if from_name:
            parts.append(f"From: {from_name}")
        if to_name:
            parts.append(f"To: {to_name}")
        if date:
            parts.append(f"Date: {date}")
        if subject:
            parts.append(f"Subject: {subject}")
        parts.append("")
        parts.append(content)
        return "\n".join(parts)

    def _position_content(
        self,
        title: str,
        company_name: str,
        description: str,
        location: str,
        started_on: str,
        finished_on: str,
    ) -> str:
        """Build content string for position."""
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if company_name:
            parts.append(f"Company: {company_name}")
        if location:
            parts.append(f"Location: {location}")
        period_parts = []
        if started_on:
            period_parts.append(f"From: {started_on}")
        if finished_on:
            period_parts.append(f"To: {finished_on}")
        if period_parts:
            parts.append(" ".join(period_parts))
        if description:
            parts.append("")
            parts.append(description)
        return "\n".join(parts)

    def _company_id(self, company_name: str) -> str:
        """Generate company ID from name."""
        return hashlib.sha1(company_name.encode("utf-8")).hexdigest()[:16]

    def _edge_id(self, source_id: str, target_ref: str, relation_type: str) -> str:
        """Generate edge ID."""
        raw = "|".join([SourceProject.LINKEDIN_ARCHIVE.value, relation_type, source_id, target_ref])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"linkedin-archive-{relation_type}-{digest}"

    def _relative_path(self, path: Path, root: Path) -> str:
        """Get relative path from root."""
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    def _tag_value(self, value: str) -> str:
        """Normalize tag value."""
        return "-".join(value.lower().split())

    def _dedupe(self, values: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output

    def _first(self, mapping: dict[str, Any], *keys: str) -> str:
        """Get first non-empty value from mapping."""
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_date(self, value: Any) -> datetime | None:
        """Parse date string in various formats."""
        text = str(value).strip() if value else ""
        if not text:
            return None

        # Try common LinkedIn date formats
        for fmt in (
            "%d %b %Y",  # 01 Jan 2024
            "%Y-%m-%d",  # 2024-01-01
            "%m/%d/%Y",  # 01/01/2024
            "%d/%m/%Y",  # 01/01/2024
        ):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        """Convert SyncState to datetime."""
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
