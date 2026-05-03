"""Adapter for vCard (VCF) contact files."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class VCardAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "vcard"

    @property
    def entity_types(self) -> list[str]:
        return ["contact"]

    def __init__(
        self,
        path: str = "",
        *,
        root_path: str = "",
        source_id_root: str | None = None,
    ) -> None:
        self.path = path or root_path
        self.source_id_root = source_id_root

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "contact" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._vcard_files(root)
        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_datetime(since) if since else None
        for file_path in files:
            stat = file_path.stat()
            file_updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if sync_at and file_updated_at <= sync_at:
                continue

            relative_path = self._relative_path(file_path, source_root)
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Parse vCard entries from the file
            vcards = self._parse_vcards(content)
            for vcard in vcards:
                unit = self._vcard_to_unit(
                    vcard,
                    relative_path,
                    file_path.stem,
                    datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                    file_updated_at
                )
                if unit:
                    result.units.append(unit)

        return result

    def _vcard_files(self, root: Path) -> list[Path]:
        if root.is_file():
            suffix = root.suffix.lower()
            return [root] if suffix in {".vcf", ".vcard"} else []
        if not root.is_dir():
            return []
        return sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".vcf", ".vcard"}
        )

    def _parse_vcards(self, content: str) -> list[dict[str, list[str]]]:
        """Parse vCard 3.0/4.0 format into a list of contact dictionaries."""
        vcards: list[dict[str, list[str]]] = []
        current_vcard: dict[str, list[str]] | None = None
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\r")

            # Handle line folding (continuation lines start with space or tab)
            while i + 1 < len(lines) and lines[i + 1] and lines[i + 1][0] in {" ", "\t"}:
                i += 1
                # Remove the leading whitespace from the continuation
                line += lines[i].rstrip("\r")[1:]

            if line.strip().upper() == "BEGIN:VCARD":
                current_vcard = {}
            elif line.strip().upper() == "END:VCARD":
                if current_vcard is not None:
                    vcards.append(current_vcard)
                current_vcard = None
            elif current_vcard is not None and ":" in line:
                # Parse property line
                prop_name, prop_value = self._parse_vcard_line(line)
                if prop_name:
                    if prop_name not in current_vcard:
                        current_vcard[prop_name] = []
                    current_vcard[prop_name].append(prop_value)

            i += 1

        return vcards

    def _parse_vcard_line(self, line: str) -> tuple[str, str]:
        """Parse a vCard property line into name and value."""
        # Split on the first colon
        if ":" not in line:
            return "", ""

        property_part, value_part = line.split(":", 1)

        # Property part may contain parameters (e.g., "TEL;TYPE=HOME")
        # We want just the property name, uppercased
        property_name = property_part.split(";")[0].strip().upper()

        # Unescape common vCard escapes
        value = value_part.replace("\\n", "\n").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")

        return property_name, value

    def _vcard_to_unit(
        self,
        vcard: dict[str, list[str]],
        source_file: str,
        file_stem: str,
        created_at: datetime,
        updated_at: datetime,
    ) -> KnowledgeUnit | None:
        """Convert a parsed vCard dictionary to a KnowledgeUnit."""
        # Extract formatted name (FN is required in vCard)
        fn_values = vcard.get("FN", [])
        if not fn_values:
            # Fallback to N (structured name) if FN is missing
            n_values = vcard.get("N", [])
            if n_values:
                # N format: "Family;Given;Middle;Prefix;Suffix"
                name_parts = n_values[0].split(";")
                # Reconstruct a reasonable full name
                fn = " ".join(part for part in [
                    name_parts[3] if len(name_parts) > 3 else "",  # prefix
                    name_parts[1] if len(name_parts) > 1 else "",  # given
                    name_parts[2] if len(name_parts) > 2 else "",  # middle
                    name_parts[0] if len(name_parts) > 0 else "",  # family
                    name_parts[4] if len(name_parts) > 4 else "",  # suffix
                ] if part).strip()
                if fn:
                    fn_values = [fn]

        title = fn_values[0] if fn_values else file_stem

        # Build metadata from vCard fields
        metadata: dict[str, str | list[str]] = {
            "source_file": source_file,
        }

        # Add common vCard fields to metadata
        for field in ["FN", "N", "ORG", "TITLE", "ROLE", "EMAIL", "TEL", "ADR", "URL", "NOTE", "BDAY", "NICKNAME", "CATEGORIES"]:
            values = vcard.get(field, [])
            if values:
                # Store as single string if only one value, otherwise as list
                metadata[field.lower()] = values[0] if len(values) == 1 else values

        # Add version if present
        version_values = vcard.get("VERSION", [])
        if version_values:
            metadata["vcard_version"] = version_values[0]

        # Build content from the most important fields
        content_parts: list[str] = []

        if fn_values:
            content_parts.append(f"Name: {fn_values[0]}")

        org_values = vcard.get("ORG", [])
        if org_values:
            content_parts.append(f"Organization: {org_values[0]}")

        title_values = vcard.get("TITLE", [])
        if title_values:
            content_parts.append(f"Title: {title_values[0]}")

        email_values = vcard.get("EMAIL", [])
        if email_values:
            if len(email_values) == 1:
                content_parts.append(f"Email: {email_values[0]}")
            else:
                content_parts.append("Emails:")
                for email in email_values:
                    content_parts.append(f"  - {email}")

        tel_values = vcard.get("TEL", [])
        if tel_values:
            if len(tel_values) == 1:
                content_parts.append(f"Phone: {tel_values[0]}")
            else:
                content_parts.append("Phones:")
                for tel in tel_values:
                    content_parts.append(f"  - {tel}")

        note_values = vcard.get("NOTE", [])
        if note_values:
            content_parts.append(f"\nNotes:\n{note_values[0]}")

        content = "\n".join(content_parts)

        # Generate source_id based on content hash for uniqueness
        source_id_base = f"{source_file}:{title}"
        digest = hashlib.sha256(source_id_base.encode("utf-8")).hexdigest()[:16]
        source_id = f"vcard:{digest}"

        return KnowledgeUnit(
            source_project=SourceProject.VCARD,
            source_id=source_id,
            source_entity_type="contact",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
