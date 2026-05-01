"""Adapter for local SRT and WebVTT transcript files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


TIMING_RE = re.compile(
    r"^\s*(?P<start>\d{2}:(?:\d{2}:)?\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:(?:\d{2}:)?\d{2}[,.]\d{3})"
)


@dataclass(frozen=True)
class Cue:
    start: str
    end: str
    text: str


class TranscriptAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "transcript"

    @property
    def entity_types(self) -> list[str]:
        return ["transcript"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "transcript" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_timestamp(since) if since else None
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in {".srt", ".vtt"}
        ):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            unit = self._read_transcript(root, path, stat.st_size, stat.st_ctime)
            if unit is not None:
                result.units.append(unit)

        return result

    def _read_transcript(
        self,
        root: Path,
        path: Path,
        file_size: int,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return None

        transcript_format = path.suffix.lower().lstrip(".")
        cues = self._parse_vtt(raw) if transcript_format == "vtt" else self._parse_srt(raw)
        if not cues:
            return None

        source_id = path.relative_to(root).as_posix()
        first_timestamp = cues[0].start
        last_timestamp = cues[-1].end
        return KnowledgeUnit(
            source_project=SourceProject.TRANSCRIPT,
            source_id=source_id,
            source_entity_type="transcript",
            title=path.stem,
            content="\n\n".join(cue.text for cue in cues),
            content_type=ContentType.ARTIFACT,
            metadata={
                "path": source_id,
                "source_path": source_id,
                "file_size": file_size,
                "transcript_format": transcript_format,
                "cue_count": len(cues),
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "duration_range": f"{first_timestamp} --> {last_timestamp}",
            },
            created_at=datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )

    def _parse_srt(self, raw: str) -> list[Cue]:
        cues: list[Cue] = []
        for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            timing_index = self._timing_line_index(lines)
            if timing_index is None:
                continue

            cue = self._cue_from_lines(lines[timing_index], lines[timing_index + 1 :])
            if cue is not None:
                cues.append(cue)
        return cues

    def _parse_vtt(self, raw: str) -> list[Cue]:
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        cues: list[Cue] = []
        index = 0
        if lines and lines[0].lstrip("\ufeff").strip().startswith("WEBVTT"):
            index = 1

        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue
            if line.startswith(("NOTE", "STYLE", "REGION")):
                index = self._skip_block(lines, index + 1)
                continue

            if "-->" in line:
                timing_line = line
                index += 1
            elif index + 1 < len(lines) and "-->" in lines[index + 1]:
                timing_line = lines[index + 1].strip()
                index += 2
            else:
                index = self._skip_block(lines, index + 1)
                continue

            text_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                text_lines.append(lines[index].strip())
                index += 1

            cue = self._cue_from_lines(timing_line, text_lines)
            if cue is not None:
                cues.append(cue)

        return cues

    def _timing_line_index(self, lines: list[str]) -> int | None:
        for index, line in enumerate(lines[:2]):
            if "-->" in line:
                return index
        return None

    def _cue_from_lines(self, timing_line: str, text_lines: list[str]) -> Cue | None:
        match = TIMING_RE.match(timing_line)
        if match is None:
            return None

        text = "\n".join(line for line in text_lines if line).strip()
        if not text:
            return None

        start = self._normalize_timestamp(match.group("start"))
        end = self._normalize_timestamp(match.group("end"))
        if start is None or end is None:
            return None
        return Cue(start=start, end=end, text=text)

    def _normalize_timestamp(self, value: str) -> str | None:
        parts = value.replace(",", ".").split(":")
        if len(parts) == 2:
            hours = 0
            minutes_text, seconds_text = parts
        elif len(parts) == 3:
            hours_text, minutes_text, seconds_text = parts
            hours = int(hours_text)
        else:
            return None

        seconds_parts = seconds_text.split(".", 1)
        if len(seconds_parts) != 2:
            return None
        minutes = int(minutes_text)
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1])
        if minutes > 59 or seconds > 59:
            return None
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _skip_block(self, lines: list[str], index: int) -> int:
        while index < len(lines) and lines[index].strip():
            index += 1
        return index

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
