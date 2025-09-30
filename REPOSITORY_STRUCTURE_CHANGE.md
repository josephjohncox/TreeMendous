# Repository Structure Change

This PR removes the `examples/` and `docs/` directories from the main Tree-Mendous repository to improve maintainability and discoverability.

## What Changed

- **Removed**: `examples/` and `docs/` directories from main repo
- **Created**: Separate repositories for better organization:
  - [tree-mendous-examples](https://github.com/josephjohncox/tree-mendous-examples)
  - [tree-mendous-docs](https://github.com/josephjohncox/tree-mendous-docs)

## For Developers

### Quick Setup
```bash
# Run once to setup development environment
./scripts/setup-dev-links.sh
```

This creates symbolic links so you can continue working as before:
```
Tree-Mendous/
├── treemendous/           # Core library
├── examples/ -> ../tree-mendous-examples/
├── docs/ -> ../tree-mendous-docs/
└── tests/
```

### Development Workflow
```bash
# Work normally (no changes!)
vim examples/new_notebook.ipynb
vim docs/new_guide.md

# Commit to respective repos
cd examples && git add . && git commit -m "Add new notebook"
cd ../docs && git add . && git commit -m "Update guide"

# Or use sync script for batch operations
./scripts/sync-repos.sh
```

## Benefits

- **🎯 Focused repositories**: Examples and docs get dedicated repos for better discoverability
- **📦 Lighter main repo**: Core library repo is smaller and more focused
- **🔍 Better SEO**: Separate repos improve GitHub search and discovery
- **👥 Specialized contributions**: Contributors can focus on specific areas
- **🚀 Same dev experience**: Unified development workflow maintained via symlinks

## Migration Guide

1. **For existing developers**: Run `./scripts/setup-dev-links.sh` once
2. **For new developers**: Clone main repo, then run setup script
3. **For users**: Clone specific repos you need (examples, docs, or core)

This change improves project organization while maintaining the excellent development experience.
