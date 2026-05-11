"""Adapter for mbox (Unix mailbox) email archive files."""

from __future__ import annotations

import hashlib
import mailbox
import re
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
SKIP_TAGS = {"script", "style"}


class _MboxHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in BLOCK_TAGS:
            self._separator()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in BLOCK_TAGS:
            self._separator()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data).strip()
        if text:
            self.parts.append(text)

    def _separator(self) -> None:
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def text(self) -> str:
        lines: list[str] = []
        current: list[str] = []
        for part in self.parts:
            if part == "\n":
                if current:
                    lines.append(" ".join(" ".join(current).split()))
                    current = []
            else:
                current.append(part)
        if current:
            lines.append(" ".join(" ".join(current).split()))
        return "\n".join(line for line in lines if line)


class MboxAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mbox"

    @property
    def entity_types(self) -> list[str]:
        return ["email"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "email" not in entity_types:
            return result

        sources = self._iter_sources()
        if not sources:
            return result

        # Track message-id to source_id mapping for threading
        message_id_map: dict[str, str] = {}
        # Store references for later edge creation
        threading_data: list[tuple[str, list[str]]] = []

        sync_at = self._sync_datetime(since) if since else None

        for source in sources:
            try:
                mbox = mailbox.mbox(source, factory=None, create=False)
            except (OSError, mailbox.Error):
                continue

            for key, message in mbox.items():
                # Extract date for sync filtering
                date_str = str(message.get("Date") or "")
                created_at = self._parse_datetime(date_str)
                if created_at is None:
                    created_at = datetime.now(timezone.utc)

                if sync_at and created_at <= sync_at:
                    continue

                # Extract message metadata
                message_id = self._clean_header(message.get("Message-ID") or "")
                from_header = self._clean_header(message.get("From") or "")
                to_header = self._clean_header(message.get("To") or "")
                cc_header = self._clean_header(message.get("Cc") or "")
                subject = self._clean_header(message.get("Subject") or "Untitled")

                # Extract body
                content = self._body_text(message)

                # Generate source_id
                source_id = self._source_id(source, key, message_id, subject)

                # Track message-id mapping
                if message_id:
                    message_id_map[message_id] = source_id

                # Extract threading headers
                in_reply_to = self._clean_header(message.get("In-Reply-To") or "")
                references = self._parse_references(message.get("References") or "")

                # Store for later edge creation
                reply_candidates = []
                if in_reply_to:
                    reply_candidates.append(in_reply_to)
                reply_candidates.extend(references)
                if reply_candidates:
                    threading_data.append((source_id, reply_candidates))

                # Create metadata
                metadata: dict = {
                    "source_file": str(source),
                    "from": from_header,
                    "to": to_header,
                    "subject": subject,
                    "date": date_str,
                }

                if cc_header:
                    metadata["cc"] = cc_header
                if message_id:
                    metadata["message_id"] = message_id
                if in_reply_to:
                    metadata["in_reply_to"] = in_reply_to
                if references:
                    metadata["references"] = references

                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.MBOX,
                        source_id=source_id,
                        source_entity_type="email",
                        title=subject,
                        content=content,
                        content_type=ContentType.ARTIFACT,
                        metadata=metadata,
                        created_at=created_at,
                    )
                )

        # Create threading edges
        emitted_edges: set[tuple[str, str]] = set()
        for source_id, reply_candidates in threading_data:
            for candidate_msg_id in reply_candidates:
                if candidate_msg_id in message_id_map:
                    parent_source_id = message_id_map[candidate_msg_id]
                    edge_key = (source_id, parent_source_id)
                    if edge_key not in emitted_edges:
                        emitted_edges.add(edge_key)
                        result.edges.append(
                            KnowledgeEdge(
                                id=self._edge_id(source_id, parent_source_id),
                                from_unit_id=source_id,
                                to_unit_id=parent_source_id,
                                relation=EdgeRelation.REPLIES_TO,
                                source=EdgeSource.SOURCE,
                                metadata={
                                    "source_project": SourceProject.MBOX.value,
                                    "from_entity_type": "email",
                                    "to_entity_type": "email",
                                },
                            )
                        )
                        # Only create one edge per email (to most recent parent)
                        break

        return result

    def _iter_sources(self) -> list[Path]:
        if not self.path:
            return []

        sources: list[Path] = []
        for raw in re.split(r"[\n,]", self.path):
            if not raw.strip():
                continue
            path = Path(raw.strip()).expanduser()
            if path.is_dir():
                sources.extend(sorted(path.rglob("*.mbox")))
                sources.extend(sorted(path.rglob("*.mbx")))
            elif path.exists() and path.is_file():
                sources.append(path)

        # Deduplicate
        deduped: list[Path] = []
        seen: set[Path] = set()
        for source in sources:
            resolved = source.resolve()
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(source)
        return deduped

    def _body_text(self, message) -> str:
        plain_parts: list[str] = []
        html_parts: list[str] = []

        for part in message.walk():
            if part.is_multipart() or part.get_content_disposition() == "attachment":
                continue

            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue

            # Use get_payload for mailbox.mboxMessage
            payload = self._decoded_payload(part)

            text = payload if isinstance(payload, str) else str(payload)
            if content_type == "text/plain":
                plain_parts.append(text.strip())
            elif content_type == "text/html":
                html_parts.append(self._strip_html(text))

        body = "\n\n".join(part for part in plain_parts if part)
        if body:
            return body
        return "\n\n".join(part for part in html_parts if part)

    def _decoded_payload(self, part) -> str:
        data = part.get_payload(decode=True)
        if not data:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return data.decode(charset, errors="replace")
        except (LookupError, UnicodeError, ValueError):
            # Unknown or invalid charset, fall back to UTF-8 with replacement
            return data.decode("utf-8", errors="replace")

    def _strip_html(self, html: str) -> str:
        parser = _MboxHTMLTextParser()
        parser.feed(html)
        parser.close()
        return parser.text()

    def _clean_header(self, value: str) -> str:
        """Clean and decode RFC 2047 encoded email headers."""
        if not value:
            return ""

        try:
            # decode_header returns a list of (decoded_bytes, charset) tuples
            decoded_parts = decode_header(value)
            result_parts = []

            for decoded_bytes, charset in decoded_parts:
                if isinstance(decoded_bytes, bytes):
                    # Decode bytes to string using specified charset or UTF-8 fallback
                    try:
                        if charset:
                            text = decoded_bytes.decode(charset, errors="replace")
                        else:
                            text = decoded_bytes.decode("utf-8", errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        # Unknown charset or decode error, fallback to UTF-8
                        text = decoded_bytes.decode("utf-8", errors="replace")
                    result_parts.append(text)
                else:
                    # Already a string
                    result_parts.append(str(decoded_bytes))

            return " ".join(result_parts).strip()
        except (ValueError, TypeError):
            # Malformed header, return as-is
            return value.strip()

    def _parse_references(self, references_header: str) -> list[str]:
        """Parse References header into list of message IDs."""
        if not references_header:
            return []
        # References are space or comma separated message IDs
        refs = re.findall(r"<[^>]+>", references_header)
        return [ref.strip() for ref in refs if ref.strip()]

    def _parse_datetime(self, date_str: str) -> datetime | None:
        if not date_str:
            return None
        try:
            parsed = parsedate_to_datetime(date_str)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    def _source_id(
        self,
        source: Path,
        key: str,
        message_id: str,
        subject: str,
    ) -> str:
        """Generate stable source ID for email."""
        # Use message-id if available, otherwise fall back to key + subject
        stable_value = message_id or f"{source.name}:{key}:{subject}"
        digest = hashlib.sha256(stable_value.encode("utf-8")).hexdigest()
        return f"mbox_{digest[:24]}"

    def _edge_id(self, from_id: str, to_id: str) -> str:
        """Generate edge ID for reply relationship."""
        raw = "|".join([SourceProject.MBOX.value, EdgeRelation.REPLIES_TO.value, from_id, to_id])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"mbox-replies-{digest[:16]}"

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
