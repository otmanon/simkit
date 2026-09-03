"""Tests for ``simkit.filesystem``.

Also a regression guard for the optional-dependency layout: only
``video_from_image_dir`` needs Pillow, so the other helpers must be importable
on a base ``numpy + scipy`` install.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from simkit.filesystem import compute_with_cache_check, get_data_directory


# --------------------------------------------------------------------------- #
# Optional-dependency layout
# --------------------------------------------------------------------------- #
def test_pil_free_helpers_are_always_available():
    """These need nothing beyond the stdlib and numpy.

    Regression test: the whole package used to sit behind the ``[video]``
    extra, so a lean install lost ``get_data_directory`` and
    ``compute_with_cache_check`` for no reason.
    """
    import simkit.filesystem as fs

    for name in ("get_data_directory", "compute_with_cache_check", "mp4_to_gif"):
        assert callable(getattr(fs, name)), name


def test_filesystem_is_reachable_from_the_top_level_package():
    import simkit

    assert simkit.filesystem.get_data_directory is get_data_directory


# --------------------------------------------------------------------------- #
# get_data_directory
# --------------------------------------------------------------------------- #
def test_get_data_directory_points_at_the_repo_data_submodule():
    path = get_data_directory()
    assert isinstance(path, str)
    assert os.path.basename(os.path.normpath(path)) == "data"
    # Resolves to a real absolute location next to the package.
    assert os.path.isabs(os.path.normpath(path))


# --------------------------------------------------------------------------- #
# compute_with_cache_check
# --------------------------------------------------------------------------- #
def test_cache_miss_computes_and_writes(tmp_path):
    cache = tmp_path / "c.npz"
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.arange(5.0)

    out = compute_with_cache_check(compute, str(cache))
    assert calls["n"] == 1
    assert cache.exists()
    assert isinstance(out, tuple) and len(out) == 1
    assert np.array_equal(out[0], np.arange(5.0))


def test_cache_hit_skips_recomputation(tmp_path):
    cache = tmp_path / "c.npz"
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.arange(5.0)

    first = compute_with_cache_check(compute, str(cache))
    second = compute_with_cache_check(compute, str(cache))
    assert calls["n"] == 1, "second call must be served from the cache"
    assert np.array_equal(first[0], second[0])


def test_read_cache_false_forces_recomputation(tmp_path):
    cache = tmp_path / "c.npz"
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.full(3, float(calls["n"]))

    compute_with_cache_check(compute, str(cache))
    out = compute_with_cache_check(compute, str(cache), read_cache=False)
    assert calls["n"] == 2
    assert np.array_equal(out[0], np.full(3, 2.0))


def test_multiple_return_values_round_trip_in_order(tmp_path):
    """Entries are stored as ``v0..vN`` and must come back in that order."""
    cache = tmp_path / "c.npz"

    def compute():
        return np.arange(3.0), np.ones((2, 2)), np.array([7.0])

    written = compute_with_cache_check(compute, str(cache))
    loaded = compute_with_cache_check(compute, str(cache))

    assert len(loaded) == 3
    for a, b in zip(written, loaded):
        assert np.array_equal(a, b)


def test_more_than_ten_values_sort_numerically_not_lexicographically(tmp_path):
    """``v10`` must follow ``v9``; a plain string sort would put it after ``v1``."""
    cache = tmp_path / "c.npz"
    n = 12

    def compute():
        return tuple(np.array([float(i)]) for i in range(n))

    compute_with_cache_check(compute, str(cache))
    loaded = compute_with_cache_check(compute, str(cache))
    assert [float(v[0]) for v in loaded] == [float(i) for i in range(n)]


def test_scalar_result_is_wrapped_in_a_tuple(tmp_path):
    cache = tmp_path / "c.npz"
    out = compute_with_cache_check(lambda: np.float64(3.5), str(cache))
    assert isinstance(out, tuple)
    assert np.isclose(float(np.asarray(out[0])), 3.5)


def test_length_one_arrays_survive_the_round_trip(tmp_path):
    """Regression: ``.item()`` used to unwrap any size-1 array to a scalar,
    so a cache hit returned a different type than a cache miss."""
    cache = tmp_path / "c.npz"

    def compute():
        return np.array([7.0]), np.array([[2.0]])

    miss = compute_with_cache_check(compute, str(cache))
    hit = compute_with_cache_check(compute, str(cache))

    for a, b in zip(miss, hit):
        assert isinstance(b, np.ndarray)
        assert a.shape == b.shape
        assert np.array_equal(a, b)


def test_non_array_objects_are_unwrapped_from_object_arrays(tmp_path):
    """A dict pickled into a 0-d object array must come back as a dict."""
    cache = tmp_path / "c.npz"

    def compute():
        return {"a": 1, "b": [2, 3]}

    compute_with_cache_check(compute, str(cache))
    (loaded,) = compute_with_cache_check(compute, str(cache))
    assert loaded == {"a": 1, "b": [2, 3]}


def test_corrupt_cache_falls_back_to_recomputation(tmp_path):
    """A truncated or unreadable cache must not crash the caller."""
    cache = tmp_path / "c.npz"
    cache.write_bytes(b"not an npz file")

    out = compute_with_cache_check(lambda: np.arange(4.0), str(cache))
    assert np.array_equal(out[0], np.arange(4.0))


# --------------------------------------------------------------------------- #
# Pillow-gated helper
# --------------------------------------------------------------------------- #
def test_video_from_image_dir_is_exported_when_pillow_is_present():
    pytest.importorskip("PIL")
    import simkit.filesystem as fs

    assert callable(fs.video_from_image_dir)
