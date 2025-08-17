## Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning where applicable.

### 2025-08-17

Added
- TODO.md with simple Now/Next/Backlog workflow and conventions.
- .github/ANNOUNCEMENT_LFS_MIGRATION.md detailing Git LFS migration steps for collaborators.

Changed
- notebooks/report.qmd refactored to use R/knitr instead of Python/Jupyter; added R setup chunk and converted code blocks (readr/knitr/htmltools).
- .gitignore updated to ignore Quarto outputs (`.quarto/`, `*_files/`, `*.knit.md`, `*.utf8.md`, rendered HTML in notebooks), R artifacts, and OS files.

Docs
- README updated with an Important Notice about Git LFS migration and quick update steps.

Chore / Infra
- Configured Git LFS for large notebooks via `.gitattributes` (tracking `notebooks/*.ipynb`).
- Migrated repository history to Git LFS for notebooks (full-history rewrite). Created safety branch `backup/pre-lfs-migrate`.

Notes (breaking)
- History was rewritten during LFS migration. Contributors must run:
	- `git lfs install`
	- `git fetch --all --tags`
	- `git reset --hard origin/main`
	- `git lfs pull`
	- Back up local work before resetting or use a backup branch.

### 2025-08-14

Removed
- Stopped tracking generated outputs: removed `data/outputs/` from Git history and added to `.gitignore` (files remain locally, no longer synced).
