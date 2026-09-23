"""Filesystem, caching, results and media helpers.

``get_data_directory``, ``compute_with_cache_check``, ``mp4_to_gif``,
``notebook_results_path``, ``save_figure`` and ``save_animation`` are
pure-stdlib/numpy at import time and always available (``save_figure`` and
``save_animation`` operate on Matplotlib objects, and ``save_animation``
imports Matplotlib's writers only when called). ``video_from_image_dir`` needs Pillow
(``pip install 'simkit[video]'``); if Pillow is missing the name is simply not
exported, so importing this package still succeeds on a lean install.
"""

from .get_data_directory import get_data_directory
from .compute_with_cache_check import compute_with_cache_check

# ffmpeg is invoked as a subprocess -- no Python dependency beyond stdlib.
from .mp4_to_gif import mp4_to_gif

# Per-notebook results folders and figure / animation saving.
from .notebook_results_path import notebook_results_path
from .save_figure import save_figure
from .save_animation import save_animation

try:  # Pillow-dependent -- pip install 'simkit[video]'
    from .video_from_image_dir import video_from_image_dir
except ImportError:  # pragma: no cover - depends on optional install
    pass
