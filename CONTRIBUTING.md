# Contributing

Thanks for your interest in contributing! This guide helps you get set up, make changes, and submit them smoothly.

## Ways to contribute
- Report bugs and request features via GitHub Issues
- Improve docs (README, guides, comments)
- Add tests, fix bugs, or implement enhancements
- Share example analyses and use cases

## Project standards at a glance
- Language: Python 3.8+; reports use Quarto with R/knitr
- License: MIT (your contributions are under the project license)
- Changelog: Keep a Changelog style in `CHANGELOG.md`
- Large files: Git LFS for `notebooks/*.ipynb`; generated outputs are ignored

## Development setup
1) Clone and environment

```bash
git clone https://github.com/hydrosolutions/exzeco-flood-risk-assessment.git
cd exzeco-flood-risk-assessment

# Recommended: Conda
conda env create -f environment.yml
conda activate exzeco
pip install -r requirements.txt

# Required once for large files
git lfs install
```

2) Optional tools
- Quarto for rendering reports (R/knitr): https://quarto.org
- R packages used in `notebooks/report.qmd`: `readr`, `knitr`, `htmltools`

## Running tests
We use `pytest`.

```bash
pytest -q
```

Add tests for new behavior (happy path + 1–2 edge cases). Keep tests fast and deterministic.

## Coding style
- Follow PEP 8; prefer type hints where practical
- Docstrings: short summary + key params/returns
- Keep functions small and focused; avoid unnecessary globals

If you use formatters/linters locally, `black`, `isort`, and `flake8` are good choices.

## Notebooks and outputs
- Large notebooks are tracked with Git LFS (`notebooks/*.ipynb`). Ensure `git lfs install` before committing/pulling.
- Don’t commit generated outputs (HTML, images, GeoTIFFs) — they are ignored via `.gitignore` (`data/outputs/`, Quarto `*_files/`, etc.).
- For Quarto documents (`.qmd`): prefer R/knitr chunks. Interactive HTML widgets render only in HTML outputs; provide text fallbacks for PDF.

## Branching and commits
- Create a topic branch from `main`: `git checkout -b feat/my-change`
- Write clear commit messages (Conventional Commits encouraged, e.g., `feat:`, `fix:`, `docs:`)
- Update `CHANGELOG.md` for user-visible changes

## Pull requests
Before opening a PR:
- [ ] Code builds and lints locally
- [ ] Tests added/updated and passing
- [ ] No large/generated files committed (respect `.gitignore` and LFS rules)
- [ ] Docs updated (README, usage, or comments) as needed
- [ ] `CHANGELOG.md` updated for noteworthy changes

PR tips:
- Keep changes focused and reviewable
- Link related Issues; describe motivation and approach
- Call out breaking changes explicitly

## Reporting issues
Please include:
- Environment info (OS, Python version, package versions)
- Steps to reproduce (minimal example if possible)
- Expected vs. actual behavior, error messages/tracebacks

## Security and responsible disclosure
If you believe you’ve found a security issue, please do not open a public Issue; instead, contact the maintainers directly.

## License
By contributing, you agree that your contributions will be licensed under the MIT License of this project.
