#!/bin/bash
# Script to update GitHub release notes for v1.0-classik-arm64-release
# Usage: ./scripts/update-release-notes.sh

set -e

# Determine repository root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="v1.0-classik-arm64-release"
REPO="Oichkatzelesfrettschen/KeenKenning"
NOTES_FILE="$SCRIPT_DIR/RELEASE_NOTES_v1.0-classik-arm64.md"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub"
    echo "Run: gh auth login"
    exit 1
fi

# Check if notes file exists
if [ ! -f "$NOTES_FILE" ]; then
    echo "Error: Release notes file not found: $NOTES_FILE"
    exit 1
fi

echo "Updating release notes for $TAG..."
echo "Repository: $REPO"
echo "Notes file: $NOTES_FILE"
echo ""

# Update the release using gh CLI
gh release edit "$TAG" \
    --repo "$REPO" \
    --notes-file "$NOTES_FILE"

echo ""
echo "✓ Release notes updated successfully!"
echo "View at: https://github.com/$REPO/releases/tag/$TAG"
