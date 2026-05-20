#!/usr/bin/env bash
# scripts/release.sh
#
# Release / docs / cleanup helper for the simkit package.
#
# Usage:
#     ./scripts/release.sh <command>
#
# Commands:
#     build          Build sdist + wheel into dist/ and run twine check
#     upload-test    Upload dist/* to TestPyPI (manual local upload)
#     test-install   Install simkit from TestPyPI in a throwaway venv
#                    and verify `import simkit`
#     upload-prod    Upload dist/* to REAL PyPI (manual, irreversible)
#     docs           Build the Sphinx HTML docs into docs/_build/html
#     docs-open      Build docs then open them in your default browser
#     clean          Remove all build/dist/docs/cache junk
#
# You can also copy/paste any single line below straight into your terminal --
# every function is just a couple of plain shell commands, no clever wrapping.
#
# First-time setup on a fresh macOS / Linux machine:
#     chmod +x scripts/release.sh
#     # Optional: drop a PyPI API token into ~/.pypirc so the upload commands
#     # don't prompt every time. See README for the format.
#
# Picking the right Python:
#     By default this script invokes whatever `python` is first on $PATH.
#     Activate your environment first (e.g. `conda activate simkit`) or set
#     the PYTHON env var to point at a specific interpreter:
#
#         PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/simkit/bin/python \
#             ./scripts/release.sh docs
#
# --------------------------------------------------------------------------
# TRADEOFFS: manual local upload vs GitHub Actions Trusted Publishing
# --------------------------------------------------------------------------
#
# LOCAL  (`upload-test` / `upload-prod` in this script)
#   + Fast iteration -- one command, no git push, no waiting on CI.
#   + Useful for the very first upload, or for one-off hotfixes.
#   - Requires a PyPI API token in ~/.pypirc on YOUR machine. Anyone with
#     your laptop can publish; tokens can leak.
#   - Build runs on YOUR box, so the wheel architecture matches your
#     interpreter. (Fine for pure-Python projects like simkit.)
#   - No verifiable link between the release artifact and a git commit.
#
# GITHUB ACTIONS  (`.github/workflows/release.yml`, triggered by a vX.Y.Z tag)
#   + No long-lived secrets. PyPI verifies the workflow run via OIDC
#     Trusted Publishing.
#   + Every release is reproducible from a tagged commit -- the build runs
#     on a clean Linux runner.
#   + Auto-publishes to TestPyPI first, then to PyPI (skipping pre-release
#     tags like v1.0.0rc1).
#   - 5-minute one-time setup on pypi.org/test.pypi.org to register the
#     repo as a Trusted Publisher and create the `pypi` / `testpypi`
#     environments in the repo settings.
#   - Slower turnaround per release (~1-2 minutes of CI).
#
# RECOMMENDATION
#     Use LOCAL for the very first upload (Trusted Publishing requires you
#     to claim the project name first). After that, switch to GITHUB
#     ACTIONS for every subsequent release by pushing a tag:
#
#         git tag v0.1.0 && git push --tags

set -euo pipefail

# Always run from the repo root, regardless of cwd.
cd "$(dirname "$0")/.."

PACKAGE="simkit"
PYTHON="${PYTHON:-python}"


cmd_clean() {
    rm -rf dist build "${PACKAGE}.egg-info" \
           docs/_build docs/autoapi \
           .pytest_cache .mypy_cache .ruff_cache
    find . -type d -name '__pycache__' -prune -exec rm -rf {} +
    find . -type f -name '*.pyc' -delete
    echo "Cleaned build / docs / cache junk."
}

cmd_build() {
    cmd_clean
    "$PYTHON" -m pip install --upgrade pip build twine
    "$PYTHON" -m build --sdist --wheel --outdir dist
    "$PYTHON" -m twine check dist/*
    echo "Built. Artifacts in dist/"
}

cmd_upload_test() {
    [ -d dist ] || cmd_build
    "$PYTHON" -m twine upload --repository testpypi dist/*
    echo
    echo "Browse:  https://test.pypi.org/project/${PACKAGE}/"
}

cmd_test_install() {
    local venv
    venv="$(mktemp -d -t simkit-test.XXXXXX)"
    echo "==> Creating throwaway venv at $venv"
    "$PYTHON" -m venv "$venv"
    "$venv/bin/python" -m pip install --upgrade pip
    echo "==> Installing $PACKAGE from TestPyPI"
    "$venv/bin/python" -m pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        "$PACKAGE"
    echo "==> Verifying import"
    "$venv/bin/python" -c "import $PACKAGE; print('OK from', $PACKAGE.__file__)"
    echo "==> Removing venv"
    rm -rf "$venv"
}

cmd_upload_prod() {
    if [ ! -d dist ]; then
        echo "No dist/ found. Run './scripts/release.sh build' first." >&2
        exit 1
    fi
    echo
    echo "  Uploading to REAL PyPI is IRREVERSIBLE."
    echo "  PyPI never lets you re-upload the same version, even after deletion."
    echo
    read -r -p "  Type 'yes' to continue: " ans
    if [ "$ans" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
    "$PYTHON" -m twine upload dist/*
    echo
    echo "Browse:  https://pypi.org/project/${PACKAGE}/"
}

cmd_docs() {
    "$PYTHON" -m sphinx -b html docs docs/_build/html
    echo "Built. Open docs/_build/html/index.html"
}

cmd_docs_open() {
    cmd_docs
    "$PYTHON" -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('docs/_build/html/index.html').resolve().as_uri())"
}

cmd_help() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}


case "${1:-help}" in
    build)        cmd_build ;;
    upload-test)  cmd_upload_test ;;
    test-install) cmd_test_install ;;
    upload-prod)  cmd_upload_prod ;;
    docs)         cmd_docs ;;
    docs-open)    cmd_docs_open ;;
    clean)        cmd_clean ;;
    help|--help|-h) cmd_help ;;
    *)
        echo "Unknown command: $1" >&2
        echo "Run './scripts/release.sh help' to see the list." >&2
        exit 2
        ;;
esac
