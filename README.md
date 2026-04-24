# Graph - Personal Knowledge Graph

A Python framework for building personal knowledge graphs with semantic search capabilities. Aggregate knowledge from multiple sources, connect related concepts, and retrieve information using full-text or AI-powered semantic search.

## Features

- **Knowledge Aggregation** - Extensible adapter pattern for ingesting data from various sources
- **Semantic Search** - AI-powered embeddings using Voyage AI or OpenAI
- **Graph Analysis** - NetworkX-based graph operations (clustering, bridges, centrality)
- **Multiple Search Modes** - Full-text, semantic, and hybrid search
- **MCP Server** - Model Context Protocol server for LLM agent integration
- **Obsidian Export** - Export your knowledge graph to Obsidian vault

## Installation

### Prerequisites

- Python 3.12 or higher
- `vault` CLI tool (for secure secrets management)
- API key from Voyage AI or OpenAI

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/graph.git
cd graph

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### With Optional AI Providers

```bash
# For Voyage AI (recommended)
pip install -e ".[voyage]"

# For OpenAI
pip install -e ".[openai]"

# For development
pip install -e ".[dev]"
```

## Configuration

### Using Vault (Recommended)

This project uses encrypted vault for API key management:

```bash
# Initialize vault if needed
vault init

# Store your API key
vault set voyage/api_key
# Enter your Voyage AI API key when prompted

# Generate .env file
vault sync .env.template
```

### Manual Configuration (Alternative)

Create a `.env` file in the project root:

```env
GRAPH_EMBEDDING_API_KEY=your_api_key_here
GRAPH_EMBEDDING_PROVIDER=voyage
GRAPH_EMBEDDING_MODEL=voyage-3-lite
```

### Getting API Keys

- **Voyage AI**: Sign up at https://www.voyageai.com
- **OpenAI**: Sign up at https://platform.openai.com

## Quick Start

### Basic Usage

```bash
# Check available commands
graph --help

# View graph statistics
graph stats

# Search knowledge
graph search "your query" --mode fulltext
graph search "your query" --mode semantic --limit 5

# Generate embeddings
graph embed --batch-size 5
```

### Knowledge Ingestion (Custom Adapters)

The included adapters are examples from the original author's personal setup. To use Graph with your own data:

1. **Create a custom adapter** by extending `BaseAdapter`:

```python
from graph.adapters.base import BaseAdapter
from graph.types.models import KnowledgeUnit, IngestResult
from datetime import datetime, timezone

class MyAdapter(BaseAdapter):
    entity_types = ["note", "document"]

    def ingest(self, since=None, entity_types=None):
        # Your ingestion logic here
        units = []
        # ... fetch and transform your data
        return IngestResult(units=units, edges=[])
```

2. **Use your adapter**:

```python
from graph.store.db import Store
from graph.config import settings

store = Store(settings.database_url)
adapter = MyAdapter()
result = adapter.ingest()
stats = store.ingest(result, "my_source")
```

See `src/graph/adapters/` for example implementations.

## Architecture

### Core Components

- **Store** (`graph.store.db`) - SQLite-based storage for units, edges, and embeddings
- **Adapters** (`graph.adapters`) - Extensible pattern for data ingestion
- **RAG Service** (`graph.rag`) - Embedding generation and semantic search
- **Graph Service** (`graph.graph`) - NetworkX-based graph analysis
- **CLI** (`graph.cli`) - Command-line interface
- **MCP Server** (`graph.mcp`) - Model Context Protocol integration

### Data Model

**KnowledgeUnit**: Core entity representing a piece of knowledge
- `id`: Unique identifier
- `title`: Human-readable title
- `content`: Main content/body
- `content_type`: Type categorization (note, task, idea, etc.)
- `source_project`: Origin project/source
- `tags`: List of tags
- `metadata`: Flexible JSON metadata
- `created_at`: Timestamp

**KnowledgeEdge**: Relationships between units
- `from_unit_id`: Source unit
- `to_unit_id`: Target unit
- `relation`: Relationship type (depends_on, references, etc.)
- `source`: How the edge was created (manual, inferred, etc.)

## AI Provider Usage

This project uses third-party AI embedding APIs for semantic search:

### Voyage AI (Default)
- **Purpose**: Generate text embeddings for semantic similarity
- **Data Sent**: Text content of knowledge units
- **Model**: `voyage-3-lite` (configurable)
- **Terms**: Users must comply with [Voyage AI Terms of Service](https://www.voyageai.com/terms)

### OpenAI (Optional)
- **Purpose**: Generate text embeddings for semantic similarity
- **Data Sent**: Text content of knowledge units
- **Model**: `text-embedding-3-small` (configurable)
- **Terms**: Users must comply with [OpenAI Terms of Service](https://openai.com/policies/terms-of-use) and [Usage Policies](https://openai.com/policies/usage-policies)

### Your Responsibilities

- **API Keys**: You are responsible for obtaining and securing your own API keys
- **Costs**: You are responsible for API usage costs
- **Data**: You control what data is sent to embedding providers
- **Compliance**: You must comply with provider terms and usage policies
- **Privacy**: Review provider data handling policies before use

### Data Privacy

- Knowledge unit content is sent to the chosen embedding provider for processing
- Embeddings are stored locally in your SQLite database
- No data is sent to embedding providers except when explicitly generating embeddings
- Check your provider's data retention and usage policies

## Intended Use

This tool is designed for **personal knowledge management**:

- ✅ Organizing personal notes, ideas, and research
- ✅ Building a personal knowledge base
- ✅ Connecting concepts across different projects
- ✅ Semantic search over personal documents
- ✅ Individual productivity and learning

## Limitations & Restrictions

This tool is **NOT designed or tested for**:

- ❌ Production/commercial applications without proper review
- ❌ Processing sensitive, confidential, or regulated data (healthcare, financial, legal)
- ❌ Multi-user or collaborative environments
- ❌ Large-scale or high-throughput applications
- ❌ Mission-critical or safety-critical systems
- ❌ Applications requiring high availability or data redundancy

**Use at your own risk.** This is alpha-stage software provided "as-is" without warranties.

## Example Adapters

The `src/graph/adapters/` directory contains example adapter implementations from the author's personal setup. These serve as:

- **Reference implementations** of the adapter pattern
- **Starting points** for your own custom adapters
- **Examples** of different data source integrations

**Note**: These adapters reference databases and configurations specific to the original author's system. They are provided as examples only and will not work without the corresponding data sources.

### Available Example Adapters

- `base.py` - Base adapter interface (extend this for custom adapters)
- `forty_two.py`, `max_adapter.py`, `presence.py`, `me.py`, `kindle.py`, `sota.py` - Example implementations

## MCP Server

Start the Model Context Protocol server for LLM agent integration:

```bash
graph serve
```

The MCP server exposes tools for:
- Searching knowledge units
- Ingesting data
- Analyzing graph structure
- Querying sync status

Configure in your MCP client (e.g., Claude Desktop) using stdio transport.

## CLI Commands

### Search & Retrieval
```bash
graph search "query" [--mode fulltext|semantic|hybrid] [--limit 10]
graph neighbors <unit-id> [--depth 1]
graph shortest-path <from-id> <to-id>
```

### Graph Analysis
```bash
graph stats
graph central [--limit 10]
graph bridges [--limit 10]
graph clusters [--min-size 3]
graph gaps [--limit 20]
graph cross-project
```

### Embeddings
```bash
graph embed [--project name] [--batch-size 5] [--delay 21.0]
```

### Export
```bash
graph export-obsidian [--vault /path/to/vault] [--folder Graph]
```

### Example-Specific Commands

These commands work with the example adapters (requires example data sources):

```bash
graph ingest [project] [--full]
graph sync-status
graph ideas [--approved] [--domain name]
graph design-briefs [--status name]
graph sync [--vault /path] [--full-ingest]
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
ruff check src/
ruff format src/
```

### Project Structure

```
graph/
├── src/graph/
│   ├── adapters/      # Data ingestion adapters (examples)
│   ├── cli/           # Command-line interface
│   ├── graph/         # Graph analysis service
│   ├── mcp/           # MCP server implementation
│   ├── rag/           # Embeddings and search
│   ├── store/         # Database layer
│   └── types/         # Data models and enums
├── tests/             # Test suite
├── .env.template      # Environment configuration template
└── README.md          # This file
```

## Extending Graph

### Creating Custom Adapters

1. Extend `BaseAdapter` class
2. Implement `ingest()` method
3. Define `entity_types` property
4. Return `IngestResult` with units and edges

### Adding Custom Edge Relations

Edit `src/graph/types/enums.py` to add new `EdgeRelation` types.

### Custom Search Filters

The CLI search supports filters:
```bash
graph search "query" --source-project name --content-type type --tag tag
```

## Troubleshooting

### Database Issues

```bash
# Reset database (warning: deletes all data)
rm graph.db
```

### Embedding Issues

```bash
# Check API key is set
vault get voyage/api_key

# Verify .env file
cat .env

# Test embedding generation
graph embed --batch-size 1 --limit 1
```

### Import Errors

```bash
# Reinstall package
pip install -e .

# Install with optional dependencies
pip install -e ".[voyage]"
```

## Contributing

Contributions are welcome! This is an open-source project under the Apache 2.0 license.

### Guidelines

- Follow existing code style (run `ruff format`)
- Add tests for new features
- Update documentation as needed
- Keep commits focused and descriptive

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

### Third-Party Dependencies

This project uses several open-source libraries. See `pyproject.toml` for the full list of dependencies and their licenses.

## Security

### Reporting Security Issues

Please report security vulnerabilities privately by creating a GitHub Security Advisory or emailing the maintainers.

### Best Practices

- Store API keys in vault, never commit them
- Review provider data policies before use
- Don't process sensitive data without proper safeguards
- Keep dependencies updated

## Disclaimer

This software is provided "as-is" without warranties of any kind, either express or implied. Users are solely responsible for:

- Compliance with AI provider terms of service
- Data privacy and security
- API usage costs
- Appropriate use of the software

See the [LICENSE](LICENSE) file for full legal details.

## Acknowledgments

Built with:
- [NetworkX](https://networkx.org/) - Graph analysis
- [Pydantic](https://pydantic.dev/) - Data validation
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol
- [Voyage AI](https://www.voyageai.com/) / [OpenAI](https://openai.com/) - Embedding APIs

---

**Status**: Alpha - Use at your own risk

**Feedback**: Issues and pull requests welcome!
