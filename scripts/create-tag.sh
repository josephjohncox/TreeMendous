#!/usr/bin/env bash
set -e

# Usage: ./scripts/create-tag.sh <type> [message]
# Example: ./scripts/create-tag.sh patch "Fix bugs"

TYPE=${1:-patch}
MESSAGE=${2:-""}

echo "🏷️  Creating $TYPE tag only..."

# Bump version
just bump-version "$TYPE"
new_version=$(just version)

# Run tests
just test

# Create release commit
release_message="Release v$new_version"
if [[ -n "$MESSAGE" ]]; then
    release_message="$release_message: $MESSAGE"
fi

git add -A
git commit -m "$release_message"
git tag -a "v$new_version" -m "$release_message"
git push origin "$(git branch --show-current)"
git push origin "v$new_version"

echo "✅ Tag v$new_version created and pushed!"
echo "🔗 Manually create release at: https://github.com/josephjohncox/Tree-Mendous/releases/tag/v$new_version"
