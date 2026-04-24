# Publication Clearance Report
**Generated**: 2026-04-24
**Repository**: /Users/taka/Project/experiments/graph
**Status**: 🔴 **BLOCKED** - Critical issues must be resolved before publication

---

## Executive Summary

This repository **CANNOT be published** in its current state. Multiple critical blockers must be addressed:

1. ❌ **No LICENSE file** - Legal requirement for open source
2. ❌ **No README.md** - Missing project documentation
3. ⚠️  **API key in .env** - Not in git history, but risky
4. ⚠️  **Hardcoded file paths** - Reveal user directory structure
5. ❌ **No AI provider terms review** - Uses Voyage AI & OpenAI APIs
6. ⚠️  **Git history contains .env.swp** - Vim swap file committed

---

## 🚨 Critical Blockers

### 1. Missing LICENSE File
**Status**: BLOCKED
**Severity**: Critical

No LICENSE file exists at repository root. This is a legal requirement for open source distribution.

**Required Action**:
- Choose an appropriate open source license (MIT, Apache 2.0, etc.)
- Add LICENSE file to repository root
- Update pyproject.toml with license field
- Verify compatibility with all dependencies

**Dependency License Summary** (from pyproject.toml):
- Core dependencies: pydantic, networkx, numpy, scipy (likely permissive)
- AI SDKs: voyageai, openai (need license verification)
- All dependencies need license compatibility review

---

### 2. Missing README.md
**Status**: BLOCKED
**Severity**: Critical

No README exists at repository root. Required for:
- Project description and purpose
- Installation instructions
- Usage examples
- API key requirements disclosure
- Intended/unsupported use cases
- AI provider usage disclosure

**Required Action**:
Create README.md that includes:
```markdown
# Graph - Personal Knowledge Graph

## Description
[What this project does]

## Requirements
- Python 3.12+
- API key from Voyage AI or OpenAI for embeddings

## Installation
[Steps]

## Configuration

### Using Vault (Recommended)
```bash
# 1. Set up vault with your API key
vault set voyage/api_key
# Enter your Voyage AI API key when prompted

# 2. Generate .env from template
vault sync .env.template
```

### Manual Setup (Alternative)
Create a `.env` file:
```env
GRAPH_EMBEDDING_API_KEY=your_api_key_here
GRAPH_EMBEDDING_PROVIDER=voyage
GRAPH_EMBEDDING_MODEL=voyage-3-lite
```

## Usage
[Examples]

## AI Provider Usage
This project uses third-party AI embedding APIs:
- Voyage AI API (default)
- OpenAI Embeddings API (optional)

Users must obtain their own API keys and comply with provider terms.

## Limitations
[Describe what this is NOT intended for]

## License
[Your chosen license]
```

---

### 3. AI Provider Terms Review
**Status**: NEEDS REVIEW
**Severity**: High

The codebase uses **two AI embedding providers**:

#### Voyage AI
- **Used by**: `VoyageEmbeddings` class (src/graph/rag/embeddings.py:19-32)
- **API Key**: Required via `GRAPH_EMBEDDING_API_KEY` environment variable
- **Model**: voyage-3-lite (default)
- **Terms Review Required**:
  - [ ] Review Voyage AI Terms of Service
  - [ ] Check data usage and retention policies
  - [ ] Verify publication/sharing permissions for outputs
  - [ ] Confirm API usage is permitted for open source projects
  - [ ] Document any attribution requirements

#### OpenAI
- **Used by**: `OpenAIEmbeddings` class (src/graph/rag/embeddings.py:35-48)
- **API Key**: Required via `GRAPH_EMBEDDING_API_KEY` (when provider=openai)
- **Model**: text-embedding-3-small (default)
- **Terms Review Required**:
  - [ ] Review OpenAI Terms of Service and Usage Policies
  - [ ] Check if API usage complies with current account type
  - [ ] Verify data handling and retention for embeddings
  - [ ] Confirm generated embeddings can be published/shared
  - [ ] Document any usage restrictions or disclaimers

**Required Actions**:
1. Review current terms for both providers
2. Document review in `.clearance/AI_PROVIDER_REVIEW.md` (keep private)
3. Add provider disclosure to README
4. Add disclaimers about user responsibility for API compliance
5. Consider adding code of conduct for acceptable use cases

---

## ⚠️  Warnings - Needs Review

### 4. Secure Secrets Management with Vault
**Status**: ✅ RESOLVED
**Severity**: Low (vault-based approach implemented)

**Finding**:
- ✅ Project now uses encrypted vault for API key storage
- ✅ `.env.template` uses vault references: `vault:voyage/api_key`
- ✅ `.env` is generated via `vault sync .env.template`
- ✅ `.env` is correctly in `.gitignore`
- ✅ `.env` itself was NOT found in git history
- ⚠️  `.env.swp` (vim swap file) WAS committed in history (minor issue)

**Git History Issue**:
```
Commit fedb878: Contains .env.swp with partial key reference
Found in branch: relay/codex/add-kindle-ingest-coverage-to-the-mcp-server-01KPXEJZ
```

**Vault Setup** (for users):
```bash
# Set up vault with your API key
vault set voyage/api_key
# Enter your Voyage AI API key when prompted

# Generate .env from template
vault sync .env.template
```

**Completed Actions**:
1. ✅ Created `.env.template` with vault references
2. ✅ Generated `.env` using vault sync
3. ✅ Added `*.swp` to .gitignore
4. ✅ `.env.template` is safe to commit (no secrets)

**Remaining Actions**:
1. Consider git history rewrite to remove .env.swp (optional)
2. Document vault setup in README

**Security Benefits**:
- API keys stored encrypted in `~/.vault/`
- `.env.template` can be safely committed
- Users manage their own secrets locally
- No plain-text secrets in repository

---

### 5. Hardcoded File Paths
**Status**: NEEDS REVIEW
**Severity**: Low

**Finding** (src/graph/config.py:14-18):
```python
forty_two_db: str = str(Path("~/Project/experiments/forty-two/forty_two.db").expanduser())
max_db: str = str(Path("~/Project/experiments/max/max.db").expanduser())
presence_db: str = str(Path("~/Project/experiments/presence/presence.db").expanduser())
me_config: str = str(Path("~/Project/experiments/me/config/projects.yaml").expanduser())
kindle_db: str = str(Path("~/Project/experiments/supabooks/supabooks.db").expanduser())
```

**Issues**:
- Reveals user directory structure (`~/Project/experiments/`)
- References to "sibling projects" (forty-two, max, presence, me, supabooks)
- May not work for other users without these projects

**Required Actions**:
1. Make paths configurable via environment variables
2. Document the "sibling projects" architecture in README
3. Provide fallback behavior when paths don't exist
4. Consider making these optional or documenting dependencies

---

### 6. Git History Cleanup
**Status**: NEEDS REVIEW
**Severity**: Low

**Finding**:
- Vim swap file (`.env.swp`) committed in history
- May contain sensitive data or be confusing to users

**Required Actions**:
1. Add `*.swp`, `*.swo`, `.*.sw?` to .gitignore
2. Consider using git-filter-repo or BFG to remove .env.swp from history
3. After cleanup, force-push to all branches (if not already public)

---

## 📋 Pre-Publication Checklist

Before making this repository public, complete ALL items:

### Legal & Licensing
- [ ] Add LICENSE file at repository root
- [ ] Update pyproject.toml with license field
- [ ] Review all dependency licenses for compatibility
- [ ] Add NOTICE file if required by dependencies
- [ ] Document third-party attributions

### Documentation
- [ ] Create comprehensive README.md
- [ ] Document AI provider requirements and terms
- [ ] Add installation instructions
- [ ] Add usage examples
- [ ] Document environment variables
- [ ] Describe intended use cases
- [ ] Describe unsupported/restricted use cases
- [ ] Add contributing guidelines (optional)

### Security & Secrets
- [ ] Revoke exposed API key in .env
- [ ] Generate new API key
- [ ] Create .env.template for documentation
- [ ] Add *.swp to .gitignore
- [ ] Consider removing .env.swp from git history
- [ ] Scan entire git history for secrets
- [ ] Remove or redact any test data with real credentials

### AI Provider Compliance
- [ ] Review Voyage AI Terms of Service (current version)
- [ ] Review OpenAI Terms of Service and Usage Policies (current version)
- [ ] Document review in .clearance/AI_PROVIDER_REVIEW.md (private)
- [ ] Add provider disclosure to README
- [ ] Add user responsibility disclaimers
- [ ] Consider usage restrictions/code of conduct

### Code Review
- [ ] Fix hardcoded paths in config.py
- [ ] Make sibling project paths configurable
- [ ] Add error handling for missing paths
- [ ] Document sibling project architecture
- [ ] Review code for any personal information
- [ ] Review test fixtures for sensitive data

### Repository Configuration
- [ ] Update .gitignore with clearance patterns
- [ ] Set up GitHub repository settings (if applicable)
- [ ] Configure branch protection (optional)
- [ ] Add GitHub Actions for CI/CD (optional)
- [ ] Add issue templates (optional)

---

## 🔄 Ongoing Monitoring

After publication, set up ongoing monitoring:

### Policy Monitoring
1. Install clearance tool: `pip install clearance` (when available)
2. Run initial monitoring: `clearance monitor --project . --report-dir clearance-report`
3. Set up scheduled checks (weekly/monthly)
4. Monitor Voyage AI and OpenAI terms changes
5. Review and update documentation when policies change

### Recommended .gitignore additions:
```gitignore
# Publication clearance (add to existing .gitignore)
clearance-report/
.clearance/AI_PROVIDER_REVIEW.md

# Vim swap files
*.swp
*.swo
.*.sw?

# Environment templates (commit these)
# .env.template
```

### Commit these monitoring files:
```
.clearance/policy-snapshots.json  # Policy baseline hashes
.clearance/README.md              # Auto-generated docs
```

---

## Summary

**Current Status**: 🔴 **BLOCKED**

**Critical Blockers**: 3
1. Missing LICENSE file
2. Missing README.md
3. No AI provider terms review

**Warnings**: 2
1. Hardcoded file paths
2. Git history cleanup needed (.env.swp)

**Resolved**: 1
1. ✅ Secrets management (now using vault)

**Estimated Effort**: 4-8 hours
- License selection and dependency review: 1-2 hours
- README creation: 1-2 hours
- AI provider terms review: 1-2 hours
- Security fixes and cleanup: 1-2 hours

**Recommendation**:
Address all critical blockers before publication. This repository contains integration with third-party AI services and references to private sibling projects, requiring careful review of terms compliance and architecture documentation.

---

## Next Steps

1. **Immediate** (Before any publication):
   - [ ] Revoke the exposed API key
   - [ ] Choose and add LICENSE file
   - [ ] Create README.md

2. **High Priority** (Required for publication):
   - [ ] Review AI provider terms
   - [ ] Document AI usage and user responsibilities
   - [ ] Fix hardcoded paths

3. **Medium Priority** (Recommended before publication):
   - [ ] Clean up git history (.env.swp removal)
   - [ ] Add .env.template
   - [ ] Full dependency license audit

4. **Post-Publication**:
   - [ ] Set up ongoing policy monitoring
   - [ ] Monitor provider terms changes
   - [ ] Keep documentation updated

---

**Report Generated By**: Claude Code (Publication Clearance Skill)
**Manual Review Required**: Yes - This is NOT legal advice
**Contact**: Consult legal counsel for licensing questions
