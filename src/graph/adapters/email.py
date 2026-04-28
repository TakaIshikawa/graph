"""Adapter for local RFC 5322 .eml email messages."""

from __future__ import annotations

from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


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


class _EmailHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


class EmailAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "email"

    @property
    def entity_types(self) -> list[str]:
        return ["email_message"]

    def __init__(self, path: str = "", source_project: str = SourceProject.EMAIL) -> None:
        self.path = path
        self.source_project = str(source_project)

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "email_message" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        sync_at = self._sync_timestamp(since) if since else None
        for path in self._email_paths(root):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            unit = self._read_email(root, path, stat.st_size, stat.st_ctime)
            if unit is not None:
                result.units.append(unit)

        return result

    def _email_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".eml" else []
        if root.is_dir():
            return sorted(
                item for item in root.rglob("*") if item.is_file() and item.suffix.lower() == ".eml"
            )
        return []

    def _read_email(
        self,
        root: Path,
        path: Path,
        file_size: int,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        try:
            with path.open("rb") as fh:
                message = BytesParser(policy=policy.default).parse(fh)
        except OSError:
            return None

        content = self._body_text(message)
        source_id = self._source_id(root, path)
        title = str(message.get("Subject") or "").strip() or path.stem
        created_at = self._created_at(str(message.get("Date") or ""), created_timestamp)

        return KnowledgeUnit(
            source_project=self.source_project,
            source_id=source_id,
            source_entity_type="email_message",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata={
                "from": str(message.get("From") or ""),
                "to": str(message.get("To") or ""),
                "cc": str(message.get("Cc") or ""),
                "date": str(message.get("Date") or ""),
                "message_id": str(message.get("Message-ID") or ""),
                "path": source_id,
                "file_size": file_size,
            },
            created_at=created_at,
        )

    def _body_text(self, message) -> str:
        plain_parts: list[str] = []
        html_parts: list[str] = []

        for part in message.walk():
            if part.is_multipart() or part.get_content_disposition() == "attachment":
                continue

            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue

            try:
                payload = part.get_content()
            except (LookupError, UnicodeDecodeError):
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
        return data.decode(charset, errors="replace")

    def _strip_html(self, html: str) -> str:
        parser = _EmailHTMLTextParser()
        parser.feed(html)
        parser.close()
        return parser.text()

    def _source_id(self, root: Path, path: Path) -> str:
        if root.is_file():
            return path.name
        return path.relative_to(root).as_posix()

    def _created_at(self, header_date: str, fallback_timestamp: float) -> datetime:
        if header_date:
            try:
                parsed = parsedate_to_datetime(header_date)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except (TypeError, ValueError):
                pass
        return datetime.fromtimestamp(fallback_timestamp, tz=timezone.utc)

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
