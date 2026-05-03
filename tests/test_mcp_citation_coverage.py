"""Focused tests for the MCP citation coverage tool."""

from __future__ import annotations

import asyncio
import json

from graph.mcp import server as mcp_server


def test_rag_citation_coverage_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "rag_citation_coverage")
    properties = tool.inputSchema["properties"]

    assert set(properties) >= {"results", "citation_keys", "url_keys"}
    assert properties["results"]["type"] == "array"
    assert properties["results"]["default"] == []
    assert properties["citation_keys"]["type"] == ["array", "string", "null"]
    assert properties["url_keys"]["type"] == ["array", "string", "null"]


def test_rag_citation_coverage_analyzes_inline_results_without_store(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: (_ for _ in ()).throw(AssertionError("store should not be opened")),
    )

    response = asyncio.run(
        mcp_server.call_tool(
            "rag_citation_coverage",
            {
                "results": [
                    {
                        "id": "with-url",
                        "title": "URL result",
                        "source": "Example",
                        "url": "https://example.com/report",
                    },
                    {
                        "id": "with-doi",
                        "title": "DOI result",
                        "metadata": {"doi": "10.1000/example"},
                    },
                    {
                        "id": "custom-citation",
                        "title": "Custom citation result",
                        "metadata": {"paper_url": "https://example.com/paper"},
                    },
                    {
                        "id": "missing",
                        "title": "Missing citation",
                        "source": "Archive",
                    },
                ],
                "url_keys": ["paper_url"],
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["total_results"] == 4
    assert payload["with_citation_count"] == 3
    assert payload["with_url_count"] == 2
    assert payload["with_identifier_count"] == 1
    assert payload["with_explicit_citation_count"] == 0
    assert payload["missing_citation_count"] == 1
    assert payload["citation_coverage_ratio"] == 0.75
    assert payload["missing_citations"] == [
        {
            "index": 3,
            "id": "missing",
            "title": "Missing citation",
            "source": "Archive",
        }
    ]
    assert payload["results"][2]["url_keys"] == ["paper_url"]
    assert payload["options"] == {
        "citation_keys": None,
        "url_keys": ["paper_url"],
    }


def test_rag_citation_coverage_empty_inputs_return_zero_metrics(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: (_ for _ in ()).throw(AssertionError("store should not be opened")),
    )

    response = asyncio.run(mcp_server.call_tool("rag_citation_coverage", {}))
    payload = json.loads(response[0].text)

    assert payload["total_results"] == 0
    assert payload["with_citation_count"] == 0
    assert payload["missing_citation_count"] == 0
    assert payload["citation_coverage_ratio"] == 0.0
    assert payload["missing_citations"] == []
    assert payload["results"] == []


def test_rag_citation_coverage_returns_structured_validation_errors(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: (_ for _ in ()).throw(AssertionError("store should not be opened")),
    )

    response = asyncio.run(
        mcp_server.call_tool(
            "rag_citation_coverage",
            {"results": [{"id": "ok"}, "bad"], "citation_keys": ["references"]},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "error": "invalid_citation_coverage_request",
        "message": "results[1] must be an object",
        "arguments": {
            "result_count": 2,
            "citation_keys": ["references"],
            "url_keys": None,
        },
    }

    bad_config = asyncio.run(
        mcp_server.call_tool(
            "rag_citation_coverage",
            {"results": [], "url_keys": [""]},
        )
    )
    bad_config_payload = json.loads(bad_config[0].text)
    assert bad_config_payload["error"] == "invalid_citation_coverage_request"
    assert (
        bad_config_payload["message"]
        == "url_keys must be a string or an array of non-empty strings"
    )
