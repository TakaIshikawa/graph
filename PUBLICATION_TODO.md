# Publication TODO - Quick Reference

**Status**: 🔴 BLOCKED - 3 critical issues
**Estimated Time**: 4-8 hours

---

## 🚨 CRITICAL - Must Complete Before Publication

### 1. Verify Vault Setup ✅ DONE
- [x] Using encrypted vault for API key storage
- [x] `.env.template` configured with vault references
- [x] `.env` generated via `vault sync .env.template`
- [ ] Document vault setup in README
- [ ] Test that application still works with vault-managed secrets

### 2. Add LICENSE File
- [ ] Choose license (MIT, Apache 2.0, GPL, etc.)
- [ ] Create `LICENSE` file at repository root
- [ ] Update `pyproject.toml` to include license field:
  ```toml
  [project]
  license = {text = "MIT"}  # or your choice
  ```
- [ ] Review dependency licenses for compatibility

### 3. Create README.md
- [ ] Copy template from `PUBLICATION_CLEARANCE_REPORT.md` (section 2)
- [ ] Fill in project description
- [ ] Add installation instructions
- [ ] Document required environment variables
- [ ] Add usage examples
- [ ] Include AI provider disclosure
- [ ] Document limitations and intended use

### 4. Review AI Provider Terms
- [ ] Open `.clearance/AI_PROVIDER_REVIEW.md`
- [ ] Review Voyage AI Terms: https://www.voyageai.com/terms
- [ ] Review OpenAI Terms: https://openai.com/policies/terms-of-use
- [ ] Review OpenAI Usage Policies: https://openai.com/policies/usage-policies
- [ ] Complete all checklist items in AI_PROVIDER_REVIEW.md
- [ ] Document any restrictions or disclaimers needed
- [ ] Update README with provider disclosures

---

## ⚠️  HIGH PRIORITY - Recommended Before Publication

### 5. Fix Hardcoded Paths
- [ ] Edit `src/graph/config.py`
- [ ] Make sibling project paths configurable via environment
- [ ] Add fallback behavior for missing paths
- [ ] Update `.env.template` with new variables
- [ ] Test with and without sibling projects
- [ ] Document sibling project architecture in README

### 6. Clean Up .gitignore
- [ ] ✅ Already added vim swap patterns
- [ ] ✅ Already added clearance patterns
- [ ] Verify `.env` is still ignored
- [ ] Verify `*.db` is still ignored

### 7. Git History Cleanup (Optional but Recommended)
- [ ] Install git-filter-repo: `brew install git-filter-repo`
- [ ] Create backup: `git clone . ../graph-backup`
- [ ] Remove .env.swp: `git filter-repo --path .env.swp --invert-paths`
- [ ] Force push to all branches (if private)
- [ ] Notify collaborators of history rewrite

---

## 📋 MEDIUM PRIORITY - Good to Have

### 8. Dependency License Audit
- [ ] List all dependencies: `pip list --format=freeze`
- [ ] Check each license for compatibility
- [ ] Document any LGPL/GPL/custom licenses
- [ ] Create NOTICE file if required
- [ ] Add third-party attributions

### 9. Additional Documentation
- [ ] Add CONTRIBUTING.md (optional)
- [ ] Add CODE_OF_CONDUCT.md (optional)
- [ ] Add issue templates (optional)
- [ ] Add pull request template (optional)

### 10. Security Review
- [ ] Scan git history for secrets: `git secrets --scan-history`
- [ ] Review test files for sensitive data
- [ ] Check logs directory is ignored
- [ ] Verify database files are ignored

---

## 🔄 POST-PUBLICATION

### 11. Set Up Monitoring
- [ ] Install clearance tool: `pip install clearance` (when available)
- [ ] Run initial check: `clearance check --project . --report-dir clearance-report`
- [ ] Run monitoring: `clearance monitor --project . --report-dir clearance-report`
- [ ] Set up scheduled checks (cron/GitHub Actions)
- [ ] Commit `.clearance/policy-snapshots.json`
- [ ] Commit `.clearance/README.md`

### 12. GitHub Repository Setup
- [ ] Create public GitHub repository
- [ ] Push code to GitHub
- [ ] Configure repository settings
- [ ] Add repository description
- [ ] Add topics/tags
- [ ] Enable GitHub Actions (optional)
- [ ] Set up branch protection (optional)

---

## Quick Start Workflow

```bash
# 1. Verify vault setup (already done)
vault get voyage/api_key  # Should return your API key
vault sync .env.template  # Regenerate .env if needed

# 2. Add LICENSE
echo "MIT License..." > LICENSE
# → Edit LICENSE with full text

# 3. Create README
cp PUBLICATION_CLEARANCE_REPORT.md README.md
# → Edit README.md with actual content

# 4. Review AI terms
# → Open .clearance/AI_PROVIDER_REVIEW.md
# → Review and complete all checklists

# 5. Fix config paths
# → Edit src/graph/config.py
# → Make paths configurable

# 6. Verify .gitignore
git status
# → Ensure no sensitive files are staged

# 7. Test everything
pytest tests/
graph --help

# 8. Final review
# → Read PUBLICATION_CLEARANCE_REPORT.md one more time
# → Verify all critical blockers resolved

# 9. Commit changes
git add .
git commit -m "Prepare repository for public release"

# 10. Push to GitHub
git remote add origin https://github.com/yourusername/graph.git
git push -u origin main
```

---

## Files Created

✅ `PUBLICATION_CLEARANCE_REPORT.md` - Full detailed report
✅ `.env.template` - Environment variable template
✅ `.clearance/AI_PROVIDER_REVIEW.md` - Provider terms review checklist
✅ Updated `.gitignore` - Added vim swaps and clearance patterns
✅ This file - Quick reference checklist

---

## Need Help?

- **Full Details**: See `PUBLICATION_CLEARANCE_REPORT.md`
- **AI Review**: See `.clearance/AI_PROVIDER_REVIEW.md`
- **Questions**: Check the release checklist at:
  `/Users/taka/.claude/skills/publication-clearance/references/release-checklist.md`

---

**Remember**: This is NOT legal advice. Consult legal counsel for licensing questions.
