# Raw commands — copy/paste into your terminal

Every block below is standalone — just paste it into bash on macOS / Linux /
WSL / Git Bash. Run them from the repo root (`cd path/to/simkit` first), with
your `simkit` environment activated (`conda activate simkit`).

If you'd rather not paste, the same commands are wrapped in
[`scripts/release.sh`](./release.sh) so you can run `./scripts/release.sh build`,
etc.

---

## One-time setup on a fresh machine

```bash
# Clone and enter the repo.
git clone https://github.com/otmanon/simkit.git
cd simkit

# Create + activate a conda env (or use venv / uv -- whatever you prefer).
conda create -n simkit python=3.11 -y
conda activate simkit

# Editable install with dev + docs tooling.
pip install -e ".[all,dev,docs]"
```

```bash
# Optional: drop your PyPI API tokens into ~/.pypirc so you're not prompted
# every upload. Get tokens at:
#   https://test.pypi.org/manage/account/token/
#   https://pypi.org/manage/account/token/
cat > ~/.pypirc <<'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEI...replace-with-real-pypi-token...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEI...replace-with-real-testpypi-token...
EOF
chmod 600 ~/.pypirc
```

---

## Day-to-day dev flow vs. cutting a release

There are two distinct flows. Most of the time you're in flow **A**.

### Flow A — You changed code but aren't ready to release

**You don't bump the version, rebuild a wheel, or touch PyPI.** Because you
installed with `pip install -e .`, every edit you make to `simkit/*.py` is
live -- `import simkit` always picks up the latest file on disk.

```bash
# 1. Edit code in simkit/.
# 2. Run the tests.
pytest

# 3. (Optional) Preview the docs.
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html        # macOS

# 4. Commit + push.
git add -A
git commit -m "Add foo() and tests"
git push
```

The only time you need to reinstall during development is if you change
`pyproject.toml` itself (added a dep, changed an extra, etc.):

```bash
pip install -e ".[all,dev,docs]"
```

### Flow B — You want users to get the change via `pip install simkit`

Now you have to bump the version. PyPI is immutable: it refuses to accept
any upload whose `name + version` already exists, even after a deletion.

```bash
# 0. Make sure main is clean and CI is green.
git checkout main
git pull
pytest

# 1. Decide the new version. Semantic versioning:
#       0.0.0 -> 0.1.0   first usable release
#       0.1.0 -> 0.1.1   bugfix only (patch)
#       0.1.0 -> 0.2.0   new features, backward compatible (minor)
#       0.x.y -> 1.0.0   stable API promise / breaking changes (major)

# 2. Edit pyproject.toml and change:
#        version = "0.0.0"
#    to:
#        version = "0.1.0"

# 3. Commit the bump on its own commit (makes the release commit greppable).
git add pyproject.toml
git commit -m "Release v0.1.0"
git push origin main

# 4. Tag the commit.
git tag v0.1.0
git push origin v0.1.0
```

What runs next depends on whether you've set up Trusted Publishing on
pypi.org / test.pypi.org:

- **If yes:** the push of `v0.1.0` triggers
  `.github/workflows/release.yml`. CI builds on a clean Linux runner,
  uploads to TestPyPI, then to PyPI. You don't type anything else.
- **If no (or you want to publish from your laptop instead):** continue
  with the manual flow in Section 7 below.

### When the release goes wrong

```text
ERROR: HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists.
```

You forgot to bump the version. Bump it in `pyproject.toml`, rebuild,
re-upload.

```bash
# You tagged the wrong commit and CI hasn't published yet:
git tag -d v0.1.0                      # delete the tag locally
git push origin :refs/tags/v0.1.0      # delete it on the remote
# fix things, then re-tag and push again
git tag v0.1.0
git push origin v0.1.0
```

After the release has been uploaded to real PyPI, you cannot recall it.
Yank it via the PyPI web UI (Settings -> Releases -> Yank, which makes pip
skip it for new installs) and release `0.1.1` with the fix.

### TL;DR

```bash
# Day-to-day:
pytest && git add -A && git commit -m "..." && git push

# Release a new version (after bumping `version` in pyproject.toml):
git tag vX.Y.Z && git push origin vX.Y.Z
```

---

## 1. Build sdist + wheel

```bash
# Clean previous artifacts (skip if dist/ doesn't exist).
rm -rf dist build simkit.egg-info

# Install build tooling and build.
python -m pip install --upgrade pip build twine
python -m build --sdist --wheel --outdir dist

# Sanity check the metadata / long description / etc.
python -m twine check dist/*
```

Result: `dist/simkit-X.Y.Z.tar.gz` and `dist/simkit-X.Y.Z-py3-none-any.whl`.

---

## 2. Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

Then browse to <https://test.pypi.org/project/simkit/> to confirm.

> If you don't have `~/.pypirc` set up, `twine` will prompt:
> - username: `__token__`
> - password: a TestPyPI API token

---

## 3. Test that install in a throwaway venv

```bash
# Make a brand-new venv outside your repo so it doesn't shadow the editable install.
VENV=$(mktemp -d -t simkit-test.XXXXXX)
python -m venv "$VENV"

# Install simkit from TestPyPI, pulling real deps (numpy/scipy) from regular PyPI.
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    simkit

# Verify it imports.
"$VENV/bin/python" -c "import simkit; print('OK from', simkit.__file__)"

# Clean up.
rm -rf "$VENV"
```

> The `--extra-index-url` is critical -- TestPyPI's mirror of `numpy` /
> `scipy` is incomplete and will otherwise fail to resolve them.

---

## 4. Upload to real PyPI

> **PyPI is immutable.** Once `simkit-0.1.0` is up, you can never re-upload
> `0.1.0`, even after deleting it. Bump `version` in `pyproject.toml`
> before each release.

```bash
python -m twine upload dist/*
```

Then browse to <https://pypi.org/project/simkit/> to confirm.

---

## 5. Build the docs locally

```bash
# Make sure docs deps are installed (already in `pip install -e ".[docs]"`).
pip install -e ".[docs]"

# Build HTML.
python -m sphinx -b html docs docs/_build/html
```

Then open the result:

```bash
# macOS
open docs/_build/html/index.html

# Linux
xdg-open docs/_build/html/index.html
```

---

## 6. Clean -- wipe every build / cache artifact

```bash
rm -rf dist build simkit.egg-info \
       docs/_build docs/autoapi \
       .pytest_cache .mypy_cache .ruff_cache
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```

---

## 7. The full release flow, end-to-end

```bash
# 0. Edit pyproject.toml -> bump `version = "0.1.0"` -> commit.
git add pyproject.toml
git commit -m "Release v0.1.0"

# 1. Clean.
rm -rf dist build simkit.egg-info

# 2. Build + check.
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*

# 3. Publish to TestPyPI.
python -m twine upload --repository testpypi dist/*

# 4. Smoke-test the install.
VENV=$(mktemp -d -t simkit-test.XXXXXX)
python -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    simkit
"$VENV/bin/python" -c "import simkit; print('OK')"
rm -rf "$VENV"

# 5. If the smoke test passed, publish to real PyPI.
python -m twine upload dist/*

# 6. Tag the release and push.
git tag v0.1.0
git push origin main --tags
```

---

## 8. Releasing via GitHub Actions instead (no laptop secrets)

After the first manual upload (step 7 above) has claimed the `simkit` name on
PyPI, every subsequent release can be a single tag push:

```bash
# Edit pyproject.toml -> bump version -> commit -> push -> tag -> push tag.
git add pyproject.toml
git commit -m "Release v0.2.0"
git push origin main

git tag v0.2.0
git push origin v0.2.0
```

That triggers `.github/workflows/release.yml`, which builds the sdist + wheel
on a clean Linux runner, uploads to TestPyPI, then to PyPI -- using OIDC
Trusted Publishing so there are no API tokens stored anywhere.

To make this work, do the one-time setup once:

1. Register the repo as a Trusted Publisher:
   - <https://test.pypi.org/manage/account/publishing/> (env name: `testpypi`)
   - <https://pypi.org/manage/account/publishing/> (env name: `pypi`)
2. In GitHub: Settings -> Environments -> create environments named
   `testpypi` and `pypi`.

Tradeoff vs. the manual flow:

|                    | Manual (steps 1–7)              | GitHub Actions (step 8)        |
| ------------------ | ------------------------------- | ------------------------------ |
| Setup cost         | 0 minutes                       | ~5 minutes, one time           |
| Per-release time   | Fast (~30 seconds)              | ~1–2 minutes of CI             |
| Secrets on laptop  | Yes (API token in `~/.pypirc`)  | None                           |
| Reproducible build | Whatever your laptop has        | Clean Linux runner, per-tag    |
| Trust trail        | "Otman ran `twine` on his Mac"  | Tag → GitHub run → PyPI (OIDC) |
| Best for           | First release, hotfixes         | Every release after the first  |
