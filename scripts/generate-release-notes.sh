#!/usr/bin/env bash
set -e

# Usage: ./scripts/generate-release-notes.sh <version> [message]
# Example: ./scripts/generate-release-notes.sh "0.1.1" "Fix bugs"

VERSION=${1}
MESSAGE=${2:-""}

if [[ -z "$VERSION" ]]; then
    echo "❌ Error: Version required"
    echo "Usage: $0 <version> [message]"
    exit 1
fi

# Generate changelog
prev_tag=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")

if [[ -n "$prev_tag" ]]; then
    changelog=$(git log --pretty=format:"- %s" "$prev_tag"..HEAD)
else
    changelog="Initial release of Tree-Mendous interval tree library"
fi

# Create release notes
release_notes_file=$(mktemp)
cat > "$release_notes_file" << EOF
# Tree-Mendous v$VERSION

$changelog

## Installation

\`\`\`bash
pip install treemendous==$VERSION
\`\`\`

## What's New

This release includes improvements to interval tree implementations, performance optimizations, and enhanced examples.
EOF

# Add custom message if provided
if [[ -n "$MESSAGE" ]]; then
    echo "" >> "$release_notes_file"
    echo "$MESSAGE" >> "$release_notes_file"
fi

# Create GitHub release
gh release create "v$VERSION" \
    --title "Release v$VERSION" \
    --notes-file "$release_notes_file" \
    --latest

# Clean up
rm "$release_notes_file"

echo "✅ GitHub release v$VERSION created with changelog!"
