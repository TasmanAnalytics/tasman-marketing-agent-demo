# Commit Message Style Guide

## Overview

This project follows a structured commit message format that emphasizes clarity, testability, and actionability. Good commit messages help future contributors understand what changed, why it changed, and how to verify it works.

## Format

```
<type>(<scope>): <subject>

<body>

Testing:
<testing instructions>

References:
<related issues or PRs>
```

## Components

### Type

Use one of these commit types:

- **feat**: New feature or significant enhancement
- **fix**: Bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, whitespace, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI/CD configuration
- **chore**: Other changes that don't modify src or test files

### Scope

Optional. Indicates what part of the codebase is affected:

- `core` - Core modules (duckdb_connector, viz, triage, etc.)
- `agents` - Agent modules
- `config` - Configuration files
- `notebook` - Jupyter notebook
- `tests` - Test suite
- `docs` - Documentation
- `build` - Build system (Makefile, pyproject.toml)

### Subject

- Use imperative mood ("Add feature" not "Added feature")
- Keep under 72 characters
- Don't end with a period
- Be specific and concise

### Body

- Wrap at 72 characters
- Explain **what** and **why**, not how (code shows how)
- Use bullet points for multiple changes
- Include motivation for the change
- Note any breaking changes

### Testing Section

Always include instructions for verifying the change:

```
Testing:
- Run `make test` to verify all tests pass
- Run `make sample-data` to generate sample database
- Execute notebook cells 1-5 to verify core functionality
- Check that `make help` shows new commands
```

### References

Link to related issues, PRs, or documentation:

```
References:
- Closes #123
- See docs/ARCHITECTURE.md for design decisions
- Related to PR #456
```

## Examples

### Example 1: Feature Addition

```
feat(core): Add template-based SQL generation engine

Implement local-first SQL generation using fuzzy template matching
and parameter filling. This reduces LLM API calls by 90% for common
queries.

Changes:
- Add LocalTextToSQL class with template matching algorithm
- Implement parameter filling (time windows, limits)
- Add entity extraction (KPIs, dimensions, time ranges)
- Include schema validation for generated SQL

Testing:
- Run `make test` - all 20 tests should pass
- Run test_local_text_to_sql.py specifically for 6 new tests
- Try "spend by channel" query in notebook - should match template
- Verify no LLM calls for template-matched queries

References:
- Implements feature described in docs/instructions-phase1.md
- Part of Phase 1 local-first architecture
```

### Example 2: Bug Fix

```
fix(core): Prevent LIMIT clause duplication in SQL queries

DuckDB connector was adding duplicate LIMIT clauses when calling
ensure_limit() on already-limited queries. This caused SQL syntax
errors in some edge cases.

Root cause: String matching for " limit " didn't catch LIMIT at
start of line after previous ensure_limit() call.

Testing:
- Run `make test` - test_limit_injection now passes
- Execute: connector.ensure_limit("SELECT * FROM t LIMIT 50")
- Verify only one LIMIT clause in output
- Check notebook queries still work correctly

References:
- Fixes test failure in tests/test_duckdb_connector.py:84
```

### Example 3: Documentation

```
docs: Add v0.1.0 state documentation and Makefile

Add comprehensive documentation describing the current state of
v0.1.0, including:
- Makefile with 10 commands for common tasks
- VERSION_0.1_STATE.md with detailed feature inventory
- CHANGELOG.md following Keep a Changelog format
- Updated README.md with version badges and production readiness

Also includes commit style guide for future contributors.

Testing:
- Run `make help` to verify all commands listed
- Run `make test` to verify test automation works
- Run `make sample-data` to verify database generation
- Review VERSION_0.1_STATE.md for accuracy
- Check README.md renders correctly on GitHub
- Verify all Makefile targets execute without errors

References:
- Establishes v0.1.0 baseline for future releases
- See VERSION_0.1_STATE.md for complete feature inventory
```

### Example 4: Refactoring

```
refactor(agents): Consolidate LLM fallback logic in base class

Extract common LLM fallback pattern from TriageAgent and
TextToSQLAgent into shared helper methods. This reduces code
duplication and makes the fallback threshold configurable.

Changes:
- No functional changes - pure refactoring
- Reduce code by ~50 lines
- Make threshold parameter consistent across agents

Testing:
- Run `make test` - all tests should still pass
- Run notebook cells - behavior should be identical
- Verify local triage still works without LLM
- Verify LLM fallback triggers at confidence < 0.6
```

### Example 5: Build System

```
build: Add Makefile for common development tasks

Add Makefile with 10 targets for installation, testing, and
running the application. This simplifies the developer experience
and provides consistent commands across environments.

Commands added:
- make install: Install dependencies with uv
- make test: Run test suite
- make run: Launch Jupyter notebook
- make clean: Remove cache files

Testing:
- Run `make help` to see all commands
- Run `make install` on clean checkout
- Run `make test` to verify pytest integration
- Run `make clean` and verify cache removal
- Check that `make run` launches notebook correctly
```

## Do's and Don'ts

### Do

- Write in imperative mood: "Add feature" not "Added feature"
- Keep subject line concise (under 72 chars)
- Include **Testing** section for all changes
- Reference related issues/PRs
- Explain **why** the change was made
- Use bullet points for readability
- Mention breaking changes prominently

### Don't

- Don't use past tense ("Added" → Use "Add")
- Don't skip the testing section
- Don't use vague descriptions ("Fix stuff", "Update things")
- Don't include obvious information code already shows
- Don't use excessive emoji (1-2 max, sparingly)
- Don't write essays - be concise but complete
- Don't commit without testing instructions

## Emoji Usage (Optional)

If using emoji, limit to 1-2 and use these meanings:

- 🎉 Initial commit or major milestone
- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation
- ♻️ Refactoring
- ⚡ Performance improvement
- ✅ Tests
- 🔧 Configuration
- 🚀 Deployment

**Important**: Emoji should enhance, not replace, clear commit messages. Many tools don't render emoji well, so messages must be clear without them.

## Why This Format?

1. **Clarity**: Future contributors understand what changed
2. **Testability**: Testing section ensures changes are verifiable
3. **Traceability**: References link to issues and design docs
4. **Consistency**: Standard format across all commits
5. **Tooling**: Compatible with changelog generators

## Enforcement

Run these checks before committing:

```bash
# 1. All tests pass
make test

# 2. Commit message is properly formatted
git log -1 --pretty=%B | head -n 1 | wc -c  # Should be < 72

# 3. Testing section is present
git log -1 --pretty=%B | grep "Testing:"

# 4. Code is clean
make clean && make test
```

## Tools

Consider using these tools to help:

- **commitizen**: Interactive commit message generator
- **commitlint**: Validate commit messages
- **conventional-changelog**: Generate changelogs from commits

## Questions?

If unsure about commit message format:
1. Look at recent commits: `git log --oneline -10`
2. Check CHANGELOG.md for examples
3. Ask in PR comments

Remember: A good commit message helps future you!
