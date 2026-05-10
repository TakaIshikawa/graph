from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.linkedin_archive import LinkedInArchiveAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_linkedin_archive_adapter_ingests_connections_with_metadata(tmp_path):
    connections_file = tmp_path / "Connections.csv"
    connections_file.write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Alice,Smith,alice@example.com,Acme Corp,Software Engineer,01 Jan 2024\n"
        "Bob,Jones,bob@example.com,Tech Inc,Product Manager,15 Feb 2024\n"
        ",,,,,\n",  # Empty row
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert [unit.source_entity_type for unit in result.units] == ["connection", "connection"]

    first = result.units[0]
    assert first.source_project == SourceProject.LINKEDIN_ARCHIVE
    assert first.source_id.startswith("linkedin_archive:connection:")
    assert first.title == "Alice Smith - Software Engineer at Acme Corp"
    assert "Name: Alice Smith" in first.content
    assert "Email: alice@example.com" in first.content
    assert "Company: Acme Corp" in first.content
    assert first.created_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert "linkedin-connection" in first.tags
    assert "company-acme-corp" in first.tags
    assert first.metadata["first_name"] == "Alice"
    assert first.metadata["last_name"] == "Smith"
    assert first.metadata["company"] == "Acme Corp"
    assert first.metadata["source_path"] == "Connections.csv"

    second = result.units[1]
    assert second.metadata["first_name"] == "Bob"
    assert second.metadata["position"] == "Product Manager"

    # Check connection edges
    assert len(result.edges) == 2
    for edge in result.edges:
        assert edge.from_unit_id == "linkedin_user:self"
        assert edge.to_unit_id in {first.source_id, second.source_id}
        assert edge.metadata["relation_type"] == "linkedin_connection"
        assert edge.relation == EdgeRelation.RELATES_TO


def test_linkedin_archive_adapter_ingests_messages_with_conversation_edges(tmp_path):
    messages_file = tmp_path / "messages.csv"
    messages_file.write_text(
        "FROM,TO,DATE,SUBJECT,CONTENT,CONVERSATION ID\n"
        "Alice Smith,Bob Jones,2024-01-01,Project Update,Let's discuss the project,conv_123\n"
        "Bob Jones,Alice Smith,2024-01-02,Re: Project Update,Sounds good,conv_123\n"
        "Alice Smith,Bob Jones,2024-01-03,Re: Project Update,When can we meet?,conv_123\n"
        "Charlie Brown,Alice Smith,2024-01-05,Hello,Nice to meet you,conv_456\n",
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 4
    assert all(unit.source_entity_type == "message" for unit in result.units)

    first = result.units[0]
    assert first.title == "Project Update"
    assert "From: Alice Smith" in first.content
    assert "To: Bob Jones" in first.content
    assert "Subject: Project Update" in first.content
    assert "Let's discuss the project" in first.content
    assert first.metadata["conversation_id"] == "conv_123"
    assert "linkedin-message" in first.tags

    # Check conversation edges
    # conv_123 has 3 messages, should create 2 edges
    # conv_456 has 1 message, no edges
    assert len(result.edges) == 2
    for edge in result.edges:
        assert edge.metadata["relation_type"] == "linkedin_conversation"
        assert edge.metadata["conversation_id"] == "conv_123"
        assert edge.relation == EdgeRelation.REFERENCES


def test_linkedin_archive_adapter_ingests_positions_with_edges(tmp_path):
    positions_file = tmp_path / "Positions.csv"
    positions_file.write_text(
        "Company Name,Title,Description,Location,Started On,Finished On\n"
        "Acme Corp,Senior Engineer,Led the team on key projects,San Francisco,01 Jan 2020,31 Dec 2022\n"
        "Tech Inc,Software Engineer,Worked on backend systems,New York,01 Feb 2018,31 Dec 2019\n"
        "StartupCo,Junior Developer,,Remote,01 Jun 2016,31 Jan 2018\n",
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 3
    assert all(unit.source_entity_type == "position" for unit in result.units)

    first = result.units[0]
    assert first.title == "Junior Developer at StartupCo"
    assert "Title: Junior Developer" in first.content
    assert "Company: StartupCo" in first.content
    assert first.created_at == datetime(2016, 6, 1, tzinfo=timezone.utc)
    assert "linkedin-position" in first.tags
    assert "company-startupco" in first.tags

    second = result.units[1]
    assert second.metadata["company_name"] == "Tech Inc"
    assert second.metadata["location"] == "New York"
    assert "Led the team" not in second.content

    third = result.units[2]
    assert "Led the team on key projects" in third.content

    # Check edges: 3 position edges (user -> position) + 3 company edges (position -> company)
    assert len(result.edges) == 6

    position_edges = [e for e in result.edges if e.metadata["relation_type"] == "linkedin_position"]
    assert len(position_edges) == 3
    for edge in position_edges:
        assert edge.from_unit_id == "linkedin_user:self"
        assert edge.relation == EdgeRelation.RELATES_TO

    company_edges = [e for e in result.edges if e.metadata["relation_type"] == "linkedin_company"]
    assert len(company_edges) == 3
    company_names = {e.metadata["company_name"] for e in company_edges}
    assert company_names == {"Acme Corp", "Tech Inc", "StartupCo"}


def test_linkedin_archive_adapter_filters_by_entity_type(tmp_path):
    (tmp_path / "Connections.csv").write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Alice,Smith,alice@example.com,Acme,Engineer,01 Jan 2024\n",
        encoding="utf-8",
    )
    (tmp_path / "messages.csv").write_text(
        "FROM,TO,DATE,SUBJECT,CONTENT,CONVERSATION ID\n"
        "Alice,Bob,2024-01-01,Hi,Hello,conv1\n",
        encoding="utf-8",
    )
    (tmp_path / "Positions.csv").write_text(
        "Company Name,Title,Description,Location,Started On,Finished On\n"
        "Acme,Engineer,Work,SF,01 Jan 2020,\n",
        encoding="utf-8",
    )

    adapter = LinkedInArchiveAdapter(path=str(tmp_path))

    connections_only = adapter.ingest(entity_types=["connection"])
    assert len(connections_only.units) == 1
    assert connections_only.units[0].source_entity_type == "connection"

    messages_only = adapter.ingest(entity_types=["message"])
    assert len(messages_only.units) == 1
    assert messages_only.units[0].source_entity_type == "message"

    positions_only = adapter.ingest(entity_types=["position"])
    assert len(positions_only.units) == 1
    assert positions_only.units[0].source_entity_type == "position"


def test_linkedin_archive_adapter_filters_by_date(tmp_path):
    (tmp_path / "Connections.csv").write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Alice,Smith,alice@example.com,Acme,Engineer,01 Jan 2024\n"
        "Bob,Jones,bob@example.com,Tech,Manager,15 Mar 2024\n",
        encoding="utf-8",
    )

    adapter = LinkedInArchiveAdapter(path=str(tmp_path))

    filtered = adapter.ingest(
        since=SyncState(
            source_project="linkedin_archive",
            source_entity_type="connection",
            last_sync_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
    )

    assert len(filtered.units) == 1
    assert filtered.units[0].metadata["first_name"] == "Bob"


def test_linkedin_archive_adapter_handles_csv_encoding_variations(tmp_path):
    # Test UTF-8 with BOM
    connections_file = tmp_path / "Connections.csv"
    connections_file.write_bytes(
        b"\xef\xbb\xbfFirst Name,Last Name,Email Address,Company,Position,Connected On\n"
        b"Alice,Smith,alice@example.com,Acme,Engineer,01 Jan 2024\n"
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["first_name"] == "Alice"


def test_linkedin_archive_adapter_handles_empty_and_invalid_files(tmp_path):
    (tmp_path / "Connections.csv").write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n",  # Header only
        encoding="utf-8",
    )
    (tmp_path / "messages.csv").write_text(
        "not,a,valid,csv\n",
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    # Should handle gracefully, no units created from invalid data
    assert len(result.units) == 0


def test_linkedin_archive_adapter_nonexistent_path():
    result = LinkedInArchiveAdapter(path="/nonexistent/path").ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_linkedin_archive_adapter_handles_various_date_formats(tmp_path):
    (tmp_path / "Connections.csv").write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Alice,Smith,alice@example.com,Acme,Engineer,01 Jan 2024\n"
        "Bob,Jones,bob@example.com,Tech,Manager,2024-02-15\n"
        "Charlie,Brown,charlie@example.com,StartupCo,Designer,03/20/2024\n",
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 3
    assert result.units[0].created_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert result.units[1].created_at == datetime(2024, 2, 15, tzinfo=timezone.utc)
    assert result.units[2].created_at == datetime(2024, 3, 20, tzinfo=timezone.utc)


def test_linkedin_archive_adapter_handles_lowercase_filenames(tmp_path):
    # Test lowercase filename variant
    (tmp_path / "connections.csv").write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Alice,Smith,alice@example.com,Acme,Engineer,01 Jan 2024\n",
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "connection"


def test_linkedin_archive_adapter_creates_edges_for_all_entity_types(tmp_path):
    (tmp_path / "Connections.csv").write_text(
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Alice,Smith,alice@example.com,Acme,Engineer,01 Jan 2024\n",
        encoding="utf-8",
    )
    (tmp_path / "Positions.csv").write_text(
        "Company Name,Title,Description,Location,Started On,Finished On\n"
        "Acme Corp,Engineer,Work,SF,01 Jan 2020,\n",
        encoding="utf-8",
    )

    result = LinkedInArchiveAdapter(path=str(tmp_path)).ingest()

    # 1 connection + 1 position = 2 units
    # 1 connection edge (user->connection) + 1 position edge (user->position) + 1 company edge (position->company) = 3 edges
    assert len(result.units) == 2
    assert len(result.edges) == 3

    edge_types = {e.metadata["relation_type"] for e in result.edges}
    assert edge_types == {"linkedin_connection", "linkedin_position", "linkedin_company"}
