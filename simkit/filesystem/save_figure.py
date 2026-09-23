import os
import shutil


def _copy_to_all(path, copy_to):
    """Copy ``path`` to each destination; a destination directory keeps the basename."""
    if copy_to is None:
        return []
    if isinstance(copy_to, (str, os.PathLike)):
        copy_to = [copy_to]
    copies = []
    for dst in copy_to:
        dst = os.fspath(dst)
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(path))
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.abspath(dst) != os.path.abspath(path):
            shutil.copyfile(path, dst)
        copies.append(dst)
    return copies


def save_figure(fig, path, copy_to=None, dpi=150, bbox_inches="tight", **savefig_kwargs):
    """Save a Matplotlib figure, creating folders, and optionally copy it elsewhere.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Output image path (format from the extension, e.g. ``.png``).
    copy_to : str or list of str, optional
        Extra destinations (file paths, or existing directories to copy into
        under the same basename). Used to mirror notebook media into ``media/``.
    dpi : int, optional
        Resolution. Defaults to 150.
    bbox_inches : str, optional
        Passed to ``savefig``. Defaults to ``"tight"``.
    **savefig_kwargs
        Forwarded to ``fig.savefig``.

    Returns
    -------
    path : str
        The primary saved path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    _copy_to_all(path, copy_to)
    return path
