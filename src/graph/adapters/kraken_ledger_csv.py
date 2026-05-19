"""Adapter for Kraken ledger CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class KrakenLedgerCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kraken_ledger_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["ledger", "transaction"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and not {"ledger", "transaction"}.intersection(entity_types):
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
        txid = first(row, "txid", "tx id", "transaction id", "transaction_id")
        refid = first(row, "refid", "ref id", "reference id", "reference_id")
        time_text = first(row, "time", "timestamp", "date", "datetime")
        timestamp = parse_datetime(time_text)
        ledger_type = first(row, "type", "ledger type")
        subtype = first(row, "subtype", "sub type")
        asset_class = first(row, "aclass", "asset class")
        asset = first(row, "asset", "currency")
        amount = parse_float(first(row, "amount"))
        fee = parse_float(first(row, "fee", "fees"))
        balance = parse_float(first(row, "balance"))
        if not any([txid, refid, time_text, ledger_type, subtype, asset_class, asset, amount is not None, fee is not None, balance is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "txid": txid,
                "refid": refid,
                "timestamp": timestamp.isoformat() if timestamp else time_text,
                "type": ledger_type,
                "subtype": subtype,
                "aclass": asset_class,
                "asset": asset,
                "amount": amount,
                "fee": fee,
                "balance": balance,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        identifier = txid or refid
        source_id = f"kraken_ledger_csv:{identifier}" if identifier else digest_source_id(
            "kraken_ledger_csv",
            time_text,
            ledger_type,
            subtype,
            asset_class,
            asset,
            amount,
            fee,
            balance,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="kraken_ledger_csv",
            source_id=source_id,
            source_entity_type="ledger",
            title=self._title(ledger_type, subtype, asset, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "crypto", "kraken", "ledger", asset, ledger_type, subtype] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _title(self, ledger_type: str, subtype: str, asset: str, amount: float | None) -> str:
        title = " ".join(part for part in [ledger_type, subtype, asset] if part) or "Kraken ledger row"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Subtype: {metadata.get('subtype')}" if metadata.get("subtype") else "",
            f"Asset class: {metadata.get('aclass')}" if metadata.get("aclass") else "",
            f"Asset: {metadata.get('asset')}" if metadata.get("asset") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Timestamp: {metadata.get('timestamp')}" if metadata.get("timestamp") else "",
            f"TxID: {metadata.get('txid')}" if metadata.get("txid") else "",
            f"RefID: {metadata.get('refid')}" if metadata.get("refid") else "",
        ]
        return "\n".join(part for part in parts if part)
