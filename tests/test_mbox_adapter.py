from __future__ import annotations

import mailbox
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.mbox import MboxAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


def write_mbox(path: Path, messages: list[dict]) -> Path:
    """Helper to create an mbox file with test messages."""
    mbox = mailbox.mbox(path, create=True)
    for msg_data in messages:
        msg = mailbox.mboxMessage()
        for header, value in msg_data.items():
            if header != "body":
                msg[header] = value
        msg.set_payload(msg_data.get("body", ""))
        mbox.add(msg)
    mbox.close()
    return path


def test_mbox_ingests_single_email_with_metadata(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Test Email",
                "Date": "Wed, 01 May 2026 10:30:00 +0000",
                "Message-ID": "<msg-001@example.com>",
                "body": "This is a test email body.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MBOX
    assert unit.source_entity_type == "email"
    assert unit.title == "Test Email"
    assert unit.content == "This is a test email body."
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.created_at == datetime(2026, 5, 1, 10, 30, tzinfo=timezone.utc)
    assert unit.metadata["from"] == "alice@example.com"
    assert unit.metadata["to"] == "bob@example.com"
    assert unit.metadata["subject"] == "Test Email"
    assert unit.metadata["message_id"] == "<msg-001@example.com>"


def test_mbox_ingests_multiple_emails(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "multiple.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "First Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<msg-001@example.com>",
                "body": "First message.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Second Email",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<msg-002@example.com>",
                "body": "Second message.",
            },
            {
                "From": "charlie@example.com",
                "To": "alice@example.com",
                "Subject": "Third Email",
                "Date": "Wed, 01 May 2026 12:00:00 +0000",
                "Message-ID": "<msg-003@example.com>",
                "body": "Third message.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 3
    titles = [unit.title for unit in result.units]
    assert "First Email" in titles
    assert "Second Email" in titles
    assert "Third Email" in titles


def test_mbox_creates_reply_edges_from_in_reply_to(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "thread.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Original Message",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<original@example.com>",
                "body": "Original message body.",
            },
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Re: Original Message",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply-001@example.com>",
                "In-Reply-To": "<original@example.com>",
                "body": "This is a reply.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    assert len(result.edges) == 1

    edge = result.edges[0]
    assert edge.relation == EdgeRelation.REPLIES_TO
    assert edge.source == EdgeSource.SOURCE

    # Verify edge connects reply to original
    original_unit = next(u for u in result.units if u.title == "Original Message")
    reply_unit = next(u for u in result.units if u.title == "Re: Original Message")
    assert edge.from_unit_id == reply_unit.source_id
    assert edge.to_unit_id == original_unit.source_id


def test_mbox_creates_reply_edges_from_references(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "thread-refs.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "team@example.com",
                "Subject": "Discussion Thread",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<thread-start@example.com>",
                "body": "Starting a discussion.",
            },
            {
                "From": "bob@example.com",
                "To": "team@example.com",
                "Subject": "Re: Discussion Thread",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply-001@example.com>",
                "References": "<thread-start@example.com>",
                "body": "My thoughts on this.",
            },
            {
                "From": "charlie@example.com",
                "To": "team@example.com",
                "Subject": "Re: Discussion Thread",
                "Date": "Wed, 01 May 2026 12:00:00 +0000",
                "Message-ID": "<reply-002@example.com>",
                "References": "<thread-start@example.com> <reply-001@example.com>",
                "body": "Building on previous comments.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 3
    assert len(result.edges) == 2

    # All edges should be REPLIES_TO
    assert all(edge.relation == EdgeRelation.REPLIES_TO for edge in result.edges)


def test_mbox_handles_email_without_message_id(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "no-msgid.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Email Without Message-ID",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "This email has no Message-ID header.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Email Without Message-ID"
    assert "message_id" not in unit.metadata


def test_mbox_handles_email_with_cc(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "with-cc.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Cc": "charlie@example.com, david@example.com",
                "Subject": "Email with CC",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<cc-test@example.com>",
                "body": "This email has CC recipients.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["cc"] == "charlie@example.com, david@example.com"


def test_mbox_handles_email_without_subject(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "no-subject.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<no-subject@example.com>",
                "body": "This email has no subject.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Untitled"


def test_mbox_filters_by_entity_type(tmp_path):
    mbox_path = write_mbox(
        tmp_path / "filter-test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Test Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "Test body.",
            }
        ],
    )

    # Should return empty when filtering by different entity type
    result = MboxAdapter(path=str(mbox_path)).ingest(entity_types=["other_type"])
    assert len(result.units) == 0

    # Should return results when filtering by email
    result = MboxAdapter(path=str(mbox_path)).ingest(entity_types=["email"])
    assert len(result.units) == 1


def test_mbox_filters_by_sync_state(tmp_path):
    from graph.types.models import SyncState

    mbox_path = write_mbox(
        tmp_path / "sync-test.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Old Email",
                "Date": "Wed, 01 Jan 2020 10:00:00 +0000",
                "Message-ID": "<old@example.com>",
                "body": "Old message.",
            },
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "New Email",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<new@example.com>",
                "body": "New message.",
            },
        ],
    )

    # Filter with sync state after old email but before new email
    sync_state = SyncState(
        source_project="mbox",
        source_entity_type="email",
        last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    result = MboxAdapter(path=str(mbox_path)).ingest(since=sync_state)

    # Should only get the new email
    assert len(result.units) == 1
    assert result.units[0].title == "New Email"


def test_mbox_handles_directory_of_mbox_files(tmp_path):
    mbox_dir = tmp_path / "mboxes"
    mbox_dir.mkdir()

    write_mbox(
        mbox_dir / "inbox.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Inbox Message",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "body": "From inbox.",
            }
        ],
    )

    write_mbox(
        mbox_dir / "sent.mbox",
        [
            {
                "From": "bob@example.com",
                "To": "alice@example.com",
                "Subject": "Sent Message",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "body": "From sent.",
            }
        ],
    )

    result = MboxAdapter(path=str(mbox_dir)).ingest()

    assert len(result.units) == 2
    titles = [unit.title for unit in result.units]
    assert "Inbox Message" in titles
    assert "Sent Message" in titles


def test_mbox_avoids_duplicate_reply_edges(tmp_path):
    """Test that only one reply edge is created even with multiple references."""
    mbox_path = write_mbox(
        tmp_path / "multi-refs.mbox",
        [
            {
                "From": "alice@example.com",
                "To": "team@example.com",
                "Subject": "Thread Start",
                "Date": "Wed, 01 May 2026 10:00:00 +0000",
                "Message-ID": "<start@example.com>",
                "body": "Starting thread.",
            },
            {
                "From": "bob@example.com",
                "To": "team@example.com",
                "Subject": "Re: Thread Start",
                "Date": "Wed, 01 May 2026 11:00:00 +0000",
                "Message-ID": "<reply@example.com>",
                "In-Reply-To": "<start@example.com>",
                "References": "<start@example.com>",
                "body": "Reply with both In-Reply-To and References.",
            },
        ],
    )

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 2
    # Should only create one edge, not duplicate
    assert len(result.edges) == 1


def test_mbox_adapter_is_registered():
    assert "mbox" in list_adapters()
    adapter = get_adapter("mbox", path="/tmp/mail.mbox")
    assert isinstance(adapter, MboxAdapter)
    assert adapter.name == "mbox"


def test_mbox_handles_html_email_content(tmp_path):
    """Test that HTML content in emails is properly stripped."""
    mbox_path = tmp_path / "html.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)
    msg = mailbox.mboxMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "HTML Email"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<html@example.com>"

    # Create multipart message with HTML
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    multipart = MIMEMultipart("alternative")
    for key in msg.keys():
        multipart[key] = msg[key]

    html_content = """
    <html>
      <body>
        <h1>Heading</h1>
        <p>This is a <strong>bold</strong> paragraph.</p>
        <ul>
          <li>Item 1</li>
          <li>Item 2</li>
        </ul>
      </body>
    </html>
    """
    html_part = MIMEText(html_content, "html")
    multipart.attach(html_part)

    mbox.add(multipart)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # HTML should be stripped to plain text
    assert "<" not in unit.content
    assert ">" not in unit.content
    assert "Heading" in unit.content
    assert "bold" in unit.content


def test_mbox_handles_multipart_with_plain_text(tmp_path):
    """Test that plain text is preferred over HTML in multipart emails."""
    mbox_path = tmp_path / "multipart.mbox"
    mbox = mailbox.mbox(mbox_path, create=True)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Multipart Email"
    msg["Date"] = "Wed, 01 May 2026 10:00:00 +0000"
    msg["Message-ID"] = "<multipart@example.com>"

    plain_text = "This is plain text content."
    html_text = "<html><body><p>This is HTML content.</p></body></html>"

    plain_part = MIMEText(plain_text, "plain")
    html_part = MIMEText(html_text, "html")

    msg.attach(plain_part)
    msg.attach(html_part)

    mbox.add(msg)
    mbox.close()

    result = MboxAdapter(path=str(mbox_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should prefer plain text over HTML
    assert unit.content == "This is plain text content."


def test_mbox_handles_empty_path():
    """Test that adapter handles empty path gracefully."""
    result = MboxAdapter(path="").ingest()
    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_mbox_handles_nonexistent_path(tmp_path):
    """Test that adapter handles nonexistent path gracefully."""
    result = MboxAdapter(path=str(tmp_path / "nonexistent.mbox")).ingest()
    assert len(result.units) == 0
    assert len(result.edges) == 0
