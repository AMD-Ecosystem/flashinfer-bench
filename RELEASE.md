# Release Process

This project uses `setuptools_scm` for automatic version management from git tags.

> **Note:** This ROCm fork does not publish to PyPI — tags only drive `setuptools_scm`
> versioning. The `build-and-upload-pypi` workflow was removed; restore it from git history
> if the fork ever needs to publish.

## Workflow

### Option 1: GitHub GUI (Recommended)

1. Go to [GitHub Releases](https://github.com/flashinfer-ai/flashinfer-bench/releases)
2. Click "Draft a new release"
3. Click "Choose a tag" → Type tag name (e.g., `v0.1.0`) → "Create new tag on publish"
4. Fill in release notes
5. Click "Publish release"

### Option 2: Command Line

```bash
# Create and push tag
git tag v0.1.0rc1
git push origin v0.1.0rc1

# Then create GitHub Release (manual)
```

## Version Format

- `v0.1.0` - Stable
- `v0.1.0rc1` - Release candidate
