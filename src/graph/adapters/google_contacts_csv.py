"""Adapter for Google Contacts CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleContactsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_contacts_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["contact", "organization", "group", "domain"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        organization_contacts: dict[str, list[KnowledgeUnit]] = {}
        organization_names: dict[str, str] = {}
        group_contacts: dict[str, list[KnowledgeUnit]] = {}
        group_names: dict[str, str] = {}
        domain_contacts: dict[str, list[KnowledgeUnit]] = {}
        contact_units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                contact_units.append(unit)
                org_name = unit.metadata.get("organization", {}).get("name")
                if org_name:
                    org_key = self._organization_key(str(org_name))
                    organization_contacts.setdefault(org_key, []).append(unit)
                    organization_names.setdefault(org_key, str(org_name))
                for group in unit.metadata.get("groups", []):
                    group_key = self._group_key(str(group))
                    if group_key:
                        group_contacts.setdefault(group_key, []).append(unit)
                        group_names.setdefault(group_key, str(group))
                for domain in self._email_domains(unit.metadata.get("emails", [])):
                    domain_contacts.setdefault(domain, []).append(unit)
                if "contact" in allowed:
                    units.append(unit)

        organization_units = [
            self._organization_unit(key, organization_names[key], contacts)
            for key, contacts in sorted(organization_contacts.items())
        ]
        if "organization" in allowed:
            units.extend(organization_units)
        group_units = [self._group_unit(key, group_names[key], contacts) for key, contacts in sorted(group_contacts.items())]
        if "group" in allowed:
            units.extend(group_units)
        domain_units = [self._domain_unit(domain, contacts) for domain, contacts in sorted(domain_contacts.items())]
        if "domain" in allowed:
            units.extend(domain_units)
        result.units.extend(sorted(units, key=lambda unit: unit.source_id))
        if {"contact", "organization"}.issubset(allowed):
            org_by_key = {self._organization_key(unit.title): unit for unit in organization_units}
            for contacts_key, contacts in organization_contacts.items():
                organization = org_by_key[contacts_key]
                for contact in contacts:
                    result.edges.append(self._organization_edge(contact, organization))
        if {"contact", "group"}.issubset(allowed):
            group_by_key = {self._group_key(unit.title): unit for unit in group_units}
            for group_key, contacts in group_contacts.items():
                group = group_by_key[group_key]
                for contact in contacts:
                    result.edges.append(self._group_edge(contact, group))
        if {"contact", "domain"}.issubset(allowed):
            domain_by_name = {unit.title: unit for unit in domain_units}
            for domain, contacts in domain_contacts.items():
                domain_unit = domain_by_name[domain]
                for contact in contacts:
                    result.edges.append(self._domain_edge(contact, domain_unit))
        if "contact" in allowed:
            result.edges.extend(self._relationship_edges(contact_units))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        if not any(str(value or "").strip() for value in row.values()):
            return None

        name = self._first(row, "Name", "Full Name", "Given Name", "Family Name")
        emails = self._repeated(row, "E-mail", "Email")
        phones = self._repeated(row, "Phone")
        addresses = self._repeated(row, "Address")
        websites = self._repeated(row, "Website")
        relationships = self._relationships(row)
        groups = self._split_groups(self._first(row, "Group Membership", "Groups"))
        organization = self._organization(row)
        birthday = self._first(row, "Birthday", "Birthdate")
        notes = self._first(row, "Notes", "Note")
        updated = self._parse_datetime(self._first(row, "Updated", "Last Modified", "Modified"))
        created_at = updated or datetime.now(timezone.utc)

        if not name:
            name = emails[0] if emails else phones[0] if phones else f"Contact {index + 1}"

        metadata = {
            "name": name,
            "names": {
                "full": name,
                "given": self._first(row, "Given Name"),
                "family": self._first(row, "Family Name"),
            },
            "given_name": self._first(row, "Given Name"),
            "family_name": self._first(row, "Family Name"),
            "notes": notes,
            "emails": emails,
            "phones": phones,
            "addresses": addresses,
            "websites": websites,
            "relationships": relationships,
            "organization": organization,
            "title": organization.get("title", ""),
            "birthday": birthday,
            "groups": groups,
            "source_file": source_file,
        }
        if updated:
            metadata["updated"] = updated.isoformat()

        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CONTACTS_CSV,
            source_id=self._source_id(row, name, emails, phones, source_file, index),
            source_entity_type="contact",
            title=name,
            content=self._content(
                name,
                notes,
                emails,
                phones,
                organization,
                addresses,
                birthday,
                groups,
                relationships,
            ),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["contact", *groups],
            created_at=created_at,
            updated_at=created_at,
        )

    def _repeated(self, row: dict[str, Any], *prefixes: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for key, value in row.items():
            key_text = str(key)
            if not any(key_text.lower().startswith(prefix.lower()) for prefix in prefixes):
                continue
            if (
                "value" not in key_text.lower()
                and "formatted" not in key_text.lower()
                and "email" not in key_text.lower()
                and key_text not in prefixes
            ):
                continue
            text = str(value or "").strip()
            identity = " ".join(text.casefold().split())
            if text and identity not in seen:
                seen.add(identity)
                values.append(text)
        return values

    def _organization(self, row: dict[str, Any]) -> dict[str, str]:
        return {
            "name": self._first(row, "Organization 1 - Name", "Organization Name", "Organization", "Company"),
            "title": self._first(row, "Organization 1 - Title", "Job Title", "Title"),
            "department": self._first(row, "Organization 1 - Department", "Department"),
        }

    def _relationships(self, row: dict[str, Any]) -> list[dict[str, str]]:
        relations: dict[str, dict[str, str]] = {}
        for key, value in row.items():
            parts = [part.strip() for part in str(key).split(" - ")]
            if len(parts) != 2 or parts[1].lower() not in {"type", "value"}:
                continue
            prefix, index = parts[0].rsplit(" ", 1) if " " in parts[0] else (parts[0], "")
            if prefix.lower() != "relation" or not index.isdigit():
                continue
            text = str(value or "").strip()
            if not text:
                continue
            relation = relations.setdefault(index, {"type": "", "value": ""})
            relation[parts[1].lower()] = " ".join(text.split())

        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index in sorted(relations, key=int):
            relation = relations[index]
            value = relation.get("value", "")
            if not value:
                continue
            item = {"type": relation.get("type", ""), "value": value}
            identity = (item["type"].casefold(), item["value"].casefold())
            if identity in seen:
                continue
            seen.add(identity)
            normalized.append(item)
        return normalized

    def _organization_unit(self, key: str, name: str, contacts: list[KnowledgeUnit]) -> KnowledgeUnit:
        departments = sorted(
            {
                str(contact.metadata.get("organization", {}).get("department"))
                for contact in contacts
                if contact.metadata.get("organization", {}).get("department")
            }
        )
        titles = sorted(
            {
                str(contact.metadata.get("organization", {}).get("title"))
                for contact in contacts
                if contact.metadata.get("organization", {}).get("title")
            }
        )
        contact_ids = sorted(contact.source_id for contact in contacts)
        email_domains = sorted(
            {
                email.rsplit("@", 1)[1].casefold()
                for contact in contacts
                for email in contact.metadata.get("emails", [])
                if "@" in email and email.rsplit("@", 1)[1].strip()
            }
        )
        source_files = sorted({str(contact.metadata.get("source_file")) for contact in contacts if contact.metadata.get("source_file")})
        content = [name, f"Contacts: {len(contacts)}"]
        if departments:
            content.append(f"Departments: {', '.join(departments)}")
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CONTACTS_CSV,
            source_id=f"google_contacts_csv:organization:{key}",
            source_entity_type="organization",
            title=name,
            content="\n".join(content),
            content_type=ContentType.METADATA,
            metadata={
                "name": name,
                "normalized_name": key,
                "contact_count": len(contacts),
                "departments": departments,
                "titles": titles,
                "contact_source_ids": contact_ids,
                "email_domains": email_domains,
                "source_files": source_files,
            },
            tags=["organization"],
            created_at=min(contact.created_at for contact in contacts),
            updated_at=max(contact.updated_at for contact in contacts),
        )

    def _organization_edge(self, contact: KnowledgeUnit, organization: KnowledgeUnit) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._edge_id(organization.source_id, contact.source_id),
            from_unit_id=organization.source_id,
            to_unit_id=contact.source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CONTACTS_CSV.value,
                "from_entity_type": "organization",
                "to_entity_type": "contact",
                "organization": organization.title,
            },
            created_at=contact.created_at,
        )

    def _group_unit(self, key: str, name: str, contacts: list[KnowledgeUnit]) -> KnowledgeUnit:
        organization_names = sorted(
            {
                str(contact.metadata.get("organization", {}).get("name"))
                for contact in contacts
                if contact.metadata.get("organization", {}).get("name")
            }
        )
        contact_ids = sorted(contact.source_id for contact in contacts)
        source_files = sorted({str(contact.metadata.get("source_file")) for contact in contacts if contact.metadata.get("source_file")})
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CONTACTS_CSV,
            source_id=f"google_contacts_csv:group:{key}",
            source_entity_type="group",
            title=name,
            content=f"Contact group: {name}\nContacts: {len(contacts)}",
            content_type=ContentType.METADATA,
            metadata={
                "name": name,
                "normalized_name": key,
                "contact_count": len(contacts),
                "contact_source_ids": contact_ids,
                "email_count": sum(len(contact.metadata.get("emails") or []) for contact in contacts),
                "organization_names": organization_names,
                "source_files": source_files,
            },
            tags=["contact-group", name],
            created_at=min(contact.created_at for contact in contacts),
            updated_at=max(contact.updated_at for contact in contacts),
        )

    def _group_edge(self, contact: KnowledgeUnit, group: KnowledgeUnit) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._group_edge_id(contact.source_id, group.source_id),
            from_unit_id=contact.source_id,
            to_unit_id=group.source_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CONTACTS_CSV.value,
                "from_entity_type": "contact",
                "to_entity_type": "group",
                "group": group.title,
            },
            created_at=contact.created_at,
        )

    def _email_domains(self, emails: list[str]) -> list[str]:
        domains: list[str] = []
        for email in emails:
            if "@" not in email:
                continue
            domain = email.rsplit("@", 1)[1].strip().strip(">").casefold()
            if domain and domain not in domains:
                domains.append(domain)
        return domains

    def _domain_unit(self, domain: str, contacts: list[KnowledgeUnit]) -> KnowledgeUnit:
        contact_ids = sorted(contact.source_id for contact in contacts)
        organizations = sorted(
            {
                str(contact.metadata.get("organization", {}).get("name"))
                for contact in contacts
                if contact.metadata.get("organization", {}).get("name")
            }
        )
        source_files = sorted({str(contact.metadata.get("source_file")) for contact in contacts if contact.metadata.get("source_file")})
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CONTACTS_CSV,
            source_id=f"google_contacts_csv:domain:{self._domain_key(domain)}",
            source_entity_type="domain",
            title=domain,
            content=f"Email domain: {domain}\nContacts: {len(contacts)}",
            content_type=ContentType.METADATA,
            metadata={
                "domain": domain,
                "contact_count": len(contacts),
                "contact_source_ids": contact_ids,
                "observed_organizations": organizations,
                "source_files": source_files,
            },
            tags=["email-domain", domain],
            created_at=min(contact.created_at for contact in contacts),
            updated_at=max(contact.updated_at for contact in contacts),
        )

    def _domain_edge(self, contact: KnowledgeUnit, domain: KnowledgeUnit) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._domain_edge_id(contact.source_id, domain.source_id),
            from_unit_id=contact.source_id,
            to_unit_id=domain.source_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CONTACTS_CSV.value,
                "from_entity_type": "contact",
                "to_entity_type": "domain",
                "domain": domain.title,
            },
            created_at=contact.created_at,
        )

    def _organization_key(self, name: str) -> str:
        return hashlib.sha256(" ".join(name.casefold().split()).encode("utf-8")).hexdigest()[:24]

    def _group_key(self, name: str) -> str:
        normalized = " ".join(name.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24] if normalized else ""

    def _domain_key(self, domain: str) -> str:
        return hashlib.sha256(domain.casefold().encode("utf-8")).hexdigest()[:24]

    def _edge_id(self, organization_id: str, contact_id: str) -> str:
        digest = hashlib.sha256(f"{organization_id}|{contact_id}|contains".encode("utf-8")).hexdigest()[:24]
        return f"google-contacts-organization-contains-{digest}"

    def _group_edge_id(self, contact_id: str, group_id: str) -> str:
        digest = hashlib.sha256(f"{contact_id}|{group_id}|references".encode("utf-8")).hexdigest()[:24]
        return f"google-contacts-group-references-{digest}"

    def _domain_edge_id(self, contact_id: str, domain_id: str) -> str:
        digest = hashlib.sha256(f"{contact_id}|{domain_id}|relates_to".encode("utf-8")).hexdigest()[:24]
        return f"google-contacts-domain-relates-{digest}"

    def _relationship_edges(self, contacts: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        contact_index: dict[str, list[KnowledgeUnit]] = {}
        for contact in contacts:
            keys = [
                self._relationship_match_key(contact.title),
                *[self._relationship_match_key(email) for email in contact.metadata.get("emails", [])],
            ]
            for key in keys:
                if key:
                    contact_index.setdefault(key, []).append(contact)

        edges: list[KnowledgeEdge] = []
        seen: set[str] = set()
        for contact in sorted(contacts, key=lambda unit: unit.source_id):
            for relationship in contact.metadata.get("relationships", []):
                value = str(relationship.get("value", "")).strip()
                match_key = self._relationship_match_key(value)
                if not match_key:
                    continue
                for related in sorted(contact_index.get(match_key, []), key=lambda unit: unit.source_id):
                    if related.source_id == contact.source_id:
                        continue
                    edge = self._relationship_edge(contact, related, str(relationship.get("type", "")), value)
                    if edge.id in seen:
                        continue
                    seen.add(edge.id)
                    edges.append(edge)
        return edges

    def _relationship_edge(
        self,
        contact: KnowledgeUnit,
        related: KnowledgeUnit,
        relationship_type: str,
        relationship_value: str,
    ) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._relationship_edge_id(contact.source_id, related.source_id, relationship_type, relationship_value),
            from_unit_id=contact.source_id,
            to_unit_id=related.source_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CONTACTS_CSV.value,
                "from_entity_type": "contact",
                "to_entity_type": "contact",
                "relationship_type": relationship_type,
                "relationship_value": relationship_value,
            },
            created_at=contact.created_at,
        )

    def _relationship_edge_id(
        self,
        contact_id: str,
        related_id: str,
        relationship_type: str,
        relationship_value: str,
    ) -> str:
        normalized_type = " ".join(relationship_type.casefold().split())
        normalized_value = self._relationship_match_key(relationship_value)
        raw = f"{contact_id}|{related_id}|{normalized_type}|{normalized_value}|references"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"google-contacts-relationship-references-{digest}"

    def _relationship_match_key(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _content(
        self,
        name: str,
        notes: str,
        emails: list[str],
        phones: list[str],
        organization: dict[str, str],
        addresses: list[str],
        birthday: str,
        groups: list[str],
        relationships: list[dict[str, str]],
    ) -> str:
        parts = [name]
        if notes:
            parts.append(notes)
        if emails:
            parts.append(f"Emails: {', '.join(emails)}")
        if phones:
            parts.append(f"Phones: {', '.join(phones)}")
        if organization.get("name"):
            parts.append(f"Organization: {organization['name']}")
        if addresses:
            parts.append(f"Addresses: {'; '.join(addresses)}")
        if birthday:
            parts.append(f"Birthday: {birthday}")
        if groups:
            parts.append(f"Groups: {', '.join(groups)}")
        if relationships:
            labels = [
                self._relationship_content_label(relationship)
                for relationship in relationships
            ]
            parts.append(f"Relationships: {', '.join(labels)}")
        return "\n".join(parts)

    def _relationship_content_label(self, relationship: dict[str, str]) -> str:
        if relationship.get("type"):
            return f"{relationship['type']}: {relationship['value']}"
        return relationship["value"]

    def _source_id(
        self,
        row: dict[str, Any],
        name: str,
        emails: list[str],
        phones: list[str],
        source_file: str,
        index: int,
    ) -> str:
        explicit = self._first(row, "ID", "Contact ID")
        identity_parts = [part for part in [",".join(sorted(email.casefold() for email in emails)), name.casefold()] if part]
        if explicit:
            raw = f"id:{explicit}"
        elif identity_parts:
            raw = "identity:" + "|".join(identity_parts)
        else:
            raw = "|".join(["row", ",".join(phones), source_file, str(index)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"google_contacts_csv:{digest}"

    def _split_groups(self, value: str) -> list[str]:
        if not value:
            return []
        cleaned = value.replace(":::*", "").replace("* myContacts", "My Contacts")
        return [item.strip() for item in cleaned.replace(";", ",").split(",") if item.strip()]

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
