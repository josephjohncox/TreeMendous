#!/usr/bin/env bash
set -e

# Usage: ./scripts/create-release.sh <type> [message]
# Example: ./scripts/create-release.sh patch "Fix Jackson network ROI"

TYPE=${1:-patch}
MESSAGE=${2:-""}

echo "🚀 Creating $TYPE release..."

# Check if working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "❌ Error: Working directory is not clean. Please commit or stash changes."
    git status --short
    exit 1
fi

# Check if we're on main branch
current_branch=$(git branch --show-current)
if [[ "$current_branch" != "main" ]]; then
    echo "⚠️  Warning: Not on main branch (currently on: $current_branch)"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Release cancelled"
        exit 1
    fi
fi

# Check if authenticated with GitHub
if ! gh auth status >/dev/null 2>&1; then
    echo "❌ Error: Not authenticated with GitHub CLI. Run: gh auth login"
    exit 1
fi

# Bump version
echo "🔢 Bumping version..."
just bump-version "$TYPE"
new_version=$(just version)

# Run tests
echo "🧪 Running tests before release..."
just test

# Build package
echo "📦 Building package..."
just build

# Create release commit
release_message="Release v$new_version"
if [[ -n "$MESSAGE" ]]; then
    release_message="$release_message: $MESSAGE"
fi

git add -A
git commit -m "$release_message"

# Create and push tag
echo "🏷️  Creating tag v$new_version..."
git tag -a "v$new_version" -m "$release_message"

echo "📤 Pushing to GitHub..."
git push origin "$current_branch"
git push origin "v$new_version"

# Generate changelog and create GitHub release
echo "📋 Creating GitHub release..."
./scripts/generate-release-notes.sh "$new_version" "$MESSAGE"

echo "✅ Release v$new_version created successfully!"
echo "🚀 GitHub release published - this will trigger CI/CD to publish to PyPI!"
