# Setup Guide

## Prerequisites

- Python 3.12 or higher
- `vault` CLI tool (for secure secrets management)
- Voyage AI API key (or OpenAI API key)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/graph.git
cd graph
```

### 2. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Configure secrets with vault

This project uses encrypted vault for API key management.

#### First-time vault setup

If you haven't used vault before:

```bash
vault init
```

#### Store your API key

```bash
vault set voyage/api_key
# Enter your Voyage AI API key when prompted
```

To get a Voyage AI API key:
1. Sign up at https://www.voyageai.com
2. Navigate to API Keys section
3. Create a new API key
4. Copy and paste it when prompted by `vault set`

#### Generate .env file

```bash
vault sync .env.template
```

This will create a `.env` file with your API key automatically resolved from the vault.

### 4. Verify installation

```bash
# Check that the CLI works
graph --help

# Verify configuration
python -c "from graph.config import settings; print(f'Provider: {settings.embedding_provider}')"
```

## Alternative: Manual Setup (Not Recommended)

If you don't want to use vault, you can manually create a `.env` file:

```env
GRAPH_EMBEDDING_API_KEY=your_api_key_here
GRAPH_EMBEDDING_PROVIDER=voyage
GRAPH_EMBEDDING_MODEL=voyage-3-lite
```

**Warning**: This stores your API key in plain text. The vault approach is more secure.

## Configuration Options

All configuration can be set via environment variables with the `GRAPH_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_EMBEDDING_API_KEY` | (required) | Your Voyage AI or OpenAI API key |
| `GRAPH_EMBEDDING_PROVIDER` | `voyage` | Embedding provider: `voyage` or `openai` |
| `GRAPH_EMBEDDING_MODEL` | `voyage-3-lite` | Model to use for embeddings |
| `GRAPH_DATABASE_URL` | `graph.db` | SQLite database location |
| `GRAPH_CONTENT_MIN_SCORE` | `7.0` | Minimum content quality score |

### Using OpenAI instead of Voyage

```bash
vault set openai/api_key
# Enter your OpenAI API key

# Update .env.template to reference openai vault key
# Then sync:
vault sync .env.template
```

Or set environment variables:
```bash
export GRAPH_EMBEDDING_PROVIDER=openai
export GRAPH_EMBEDDING_MODEL=text-embedding-3-small
```

## Running Tests

```bash
pytest tests/
```

## Troubleshooting

### "vault: command not found"

Install the vault CLI tool. See your project's global vault documentation.

### "GRAPH_EMBEDDING_API_KEY not set"

Make sure you've run:
```bash
vault set voyage/api_key
vault sync .env.template
```

### "No module named 'voyageai'"

Install optional dependencies:
```bash
pip install -e ".[voyage]"
```

For OpenAI:
```bash
pip install -e ".[openai]"
```

## Security Notes

- **Never commit `.env` files** - They're in `.gitignore` for security
- **Use vault for API keys** - Keys are encrypted at rest in `~/.vault/`
- **`.env.template` is safe to commit** - It only contains vault references
- **Rotate API keys regularly** - Update via `vault set voyage/api_key`

## Next Steps

- Read the [README](README.md) for usage examples
- Check [API documentation](docs/) (if available)
- Review [contributing guidelines](CONTRIBUTING.md) (if available)
