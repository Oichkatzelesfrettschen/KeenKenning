# Release Notes Update Instructions

## Problem
The release notes for "Keen Classik ARM64 Release - 2026-01-05" (tag `v1.0-classik-arm64-release`) contained internal QA language that sounded like self-congratulatory technical reports rather than user-facing release notes.

## Solution
Created new user-focused release notes that:
- Explain what the app actually does for users
- Highlight key features without marketing jargon
- Provide clear download and installation instructions
- Remove internal QA metrics (e.g., "ZERO defects", "100% pass rate")

## Files Created

1. **RELEASE_NOTES_v1.0-classik-arm64.md** - New release notes in Markdown format
2. **scripts/update-release-notes.sh** - Automated script to update the GitHub release

## How to Update the Release

### Option 1: Using the Script (Recommended)

If you have the GitHub CLI (`gh`) installed and authenticated:

```bash
./scripts/update-release-notes.sh
```

The script is already executable and will:
- Check that `gh` CLI is installed
- Verify you're authenticated with GitHub (run `gh auth login` if not)
- Check that the release notes file exists
- Update the release on GitHub

This will automatically update the release notes on GitHub.

### Option 2: Using GitHub CLI Manually

```bash
gh release edit v1.0-classik-arm64-release \
    --repo Oichkatzelesfrettschen/KeenKenning \
    --notes-file RELEASE_NOTES_v1.0-classik-arm64.md
```

### Option 3: Via GitHub Web Interface

1. Go to https://github.com/Oichkatzelesfrettschen/KeenKenning/releases/tag/v1.0-classik-arm64-release
2. Click "Edit release"
3. Copy the contents of `RELEASE_NOTES_v1.0-classik-arm64.md`
4. Paste into the release description field
5. Click "Update release"

## Key Changes

### Before (Self-Congratulatory)
- "Production release with full ASCII compliance, comprehensive testing..."
- "Build: SUCCESSFUL (zero warnings, -Werror enforced)"
- "Tests: 70/70 PASS"
- "Defects: ZERO found"
- "Regressions: ZERO detected"
- "Production-ready for immediate deployment"

### After (User-Focused)
- "A traditional KenKen-style logic puzzle game for Android"
- Focuses on what users get: puzzle gameplay, difficulty levels, save features
- Clear installation instructions
- Honest about it being a debug build
- Mentions what's coming next (Keen Kenning variant)

## Notes

The new release notes are:
- Written from the user's perspective
- Focus on features and how to use the app
- Include practical information (system requirements, installation)
- Avoid technical QA jargon
- Are honest about limitations (debug build, performance considerations)
