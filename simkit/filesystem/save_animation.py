import os

from .save_figure import _copy_to_all


def save_animation(anim, path, fps=20, copy_to=None, dpi=None, bitrate=2400):
    """Save a Matplotlib animation to ``.mp4`` (ffmpeg) or ``.gif`` (Pillow).

    Parameters
    ----------
    anim : matplotlib.animation.Animation
        Animation to write, e.g. a ``FuncAnimation``.
    path : str
        Output path; the extension picks the writer (``.gif`` -> Pillow,
        anything else -> ffmpeg).
    fps : int, optional
        Frames per second. Defaults to 20.
    copy_to : str or list of str, optional
        Extra destinations (file paths, or existing directories to copy into
        under the same basename). Used to mirror notebook media into ``media/``.
    dpi : int, optional
        Frame resolution; Matplotlib's default when omitted.
    bitrate : int, optional
        ffmpeg bitrate in kbps; defaults to 2400. Ignored for ``.gif``.

    Returns
    -------
    path : str
        The primary saved path.
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    anim.save(path, writer=writer, dpi=dpi)
    _copy_to_all(path, copy_to)
    return path
