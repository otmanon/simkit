# Release / docs / cleanup helper for the simkit package.
#
# Same commands as scripts/release.ps1, for Linux / macOS / WSL / Git Bash.
# Run `make help` to see the list.
#
# --------------------------------------------------------------------------
# TRADEOFFS: manual local upload vs GitHub Actions Trusted Publishing
# --------------------------------------------------------------------------
#
# LOCAL (`make upload-test` / `make upload-pypi` from this file)
#   + Fast iteration -- one command, no git push, no waiting on CI.
#   + Useful for the very first upload or for one-off hotfixes.
#   - Requires a PyPI API token in ~/.pypirc on YOUR machine. Anyone with
#     your laptop can publish; tokens can leak.
#   - Build runs on YOUR box, so the wheel architecture matches your
#     interpreter. (Fine for pure-Python projects like simkit.)
#   - No verifiable link between the release artifact and a git commit.
#
# GITHUB ACTIONS (`.github/workflows/release.yml`, triggered by a vX.Y.Z tag)
#   + No long-lived secrets. PyPI verifies the workflow run via OIDC
#     Trusted Publishing.
#   + Every release is reproducible from a tagged commit -- the build runs
#     on a clean Linux runner.
#   + Auto-publishes to TestPyPI first, then to PyPI (skipping pre-release
#     tags like v1.0.0rc1).
#   - 5-minute one-time setup on pypi.org/test.pypi.org to register the repo
#     as a Trusted Publisher and create the `pypi` / `testpypi` environments
#     in the repo settings.
#   - Slower turnaround per release (~1-2 minutes of CI).
#
# RECOMMENDATION
#   Use LOCAL for the very first upload (Trusted Publishing requires you to
#   claim the project name first). After that, switch to GITHUB ACTIONS for
#   every subsequent release by pushing a tag.

PYTHON  ?= python
PACKAGE := simkit

.PHONY: help build upload-test test-install upload-pypi docs docs-open clean

help:
	@echo "Targets:"
	@echo "  build         Build sdist + wheel into dist/ and run twine check"
	@echo "  upload-test   Upload dist/* to TestPyPI (manual local upload)"
	@echo "  test-install  Install simkit from TestPyPI in a throwaway venv"
	@echo "  upload-pypi   Upload dist/* to REAL PyPI (manual, irreversible)"
	@echo "  docs          Build the Sphinx HTML docs"
	@echo "  docs-open     Build docs then open them in your default browser"
	@echo "  clean         Remove all build/dist/docs/cache junk"

build: clean
	$(PYTHON) -m pip install --upgrade --quiet pip build twine
	$(PYTHON) -m build --sdist --wheel --outdir dist
	$(PYTHON) -m twine check dist/*

upload-test: build
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo ""
	@echo "Browse:  https://test.pypi.org/project/$(PACKAGE)/"

test-install:
	@VENV=$$(mktemp -d /tmp/simkit-test-XXXXXX) && \
		$(PYTHON) -m venv $$VENV && \
		$$VENV/bin/python -m pip install --upgrade --quiet pip && \
		$$VENV/bin/python -m pip install \
			--index-url https://test.pypi.org/simple/ \
			--extra-index-url https://pypi.org/simple/ \
			$(PACKAGE) && \
		$$VENV/bin/python -c "import $(PACKAGE); print('OK from', $(PACKAGE).__file__)" && \
		rm -rf $$VENV

upload-pypi:
	@echo ""
	@echo "  Uploading to REAL PyPI is IRREVERSIBLE."
	@echo "  PyPI never lets you re-upload the same version, even after deletion."
	@echo ""
	@read -p "  Type 'yes' to continue: " ans && [ "$$ans" = "yes" ] || (echo "Aborted." && exit 1)
	$(PYTHON) -m twine upload dist/*
	@echo ""
	@echo "Browse:  https://pypi.org/project/$(PACKAGE)/"

docs:
	$(PYTHON) -m sphinx -b html docs docs/_build/html

docs-open: docs
	$(PYTHON) -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('docs/_build/html/index.html').resolve().as_uri())"

clean:
	rm -rf dist build *.egg-info docs/_build docs/autoapi .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
