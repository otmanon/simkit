import numpy as np
import os


def compute_with_cache_check(func, cache_path, read_cache=True):
    """Compute a function's result with optional caching.

    Either loads previously computed results from an ``.npz`` cache file, or
    computes them from scratch and saves them to that file.

    Parameters
    ----------
    func : callable
        Zero-argument function producing the results. May return a single value
        or a tuple of values.
    cache_path : str
        Path to the cache file (``.npz`` format).
    read_cache : bool, optional
        Whether to attempt reading from the cache. Defaults to True. When
        False, ``func`` always runs and the cache is overwritten.

    Returns
    -------
    tuple
        The computed results, always as a tuple -- a single return value is
        wrapped in a length-1 tuple.

    Notes
    -----
    Values are stored under the keys ``v0 .. vN`` and read back in numeric
    order, so the tuple ordering survives a round trip past ten entries.
    Non-array objects (dicts, lists) are pickled by ``np.savez`` into 0-d
    object arrays and unwrapped with ``.item()`` on load; genuine arrays --
    including length-1 arrays -- come back unchanged.
    """
    if read_cache:
        try:
            output = _load(cache_path)
        except Exception:
            print("Could not read from cache. Recomputing from scratch...")
            output = _compute_and_save(func, cache_path)
    else:
        print("Will not read from cache. Recomputing from scratch...")
        output = _compute_and_save(func, cache_path)

    if not isinstance(output, tuple):
        output = (output,)
    return output


def _load(cache_path):
    """Read a cache file back into the tuple that was written."""
    data = np.load(cache_path, allow_pickle=True)
    output = ()
    for k in sorted(data.files, key=lambda s: int(s[1:])):
        v = data[k]
        # Only 0-d arrays are unwrapped: those are the pickled non-array
        # objects. Unwrapping any size-1 array would silently turn a stored
        # ``np.array([7.0])`` into the float ``7.0`` on the cache-hit path.
        output += (v.item() if v.ndim == 0 else v,)
    return output


def _compute_and_save(func, cache_path):
    output = func()
    if not isinstance(output, tuple):
        output = (output,)
    np.savez(cache_path, **{f"v{i}": arg for i, arg in enumerate(output)})
    return output
