import os


def notebook_results_path(notebook_name, filename=None, root="results", mkdir=True):
    """Path to a notebook's private results folder, ``<root>/<notebook_name>/``.

    Every file a notebook writes (plots, videos, caches) lives under one folder
    named after the notebook, so outputs from different notebooks never collide.

    Parameters
    ----------
    notebook_name : str
        Notebook name without extension, e.g. ``"004_elastostatics_minimization"``.
    filename : str, optional
        File inside the folder. When given, the full file path is returned.
    root : str, optional
        Parent directory of all results folders. Defaults to ``"results"``,
        relative to the current working directory.
    mkdir : bool, optional
        Create the folder if it does not exist. Defaults to True.

    Returns
    -------
    path : str
        ``<root>/<notebook_name>`` or ``<root>/<notebook_name>/<filename>``.
    """
    folder = os.path.join(root, notebook_name)
    if mkdir:
        os.makedirs(folder, exist_ok=True)
    if filename is None:
        return folder
    return os.path.join(folder, filename)
