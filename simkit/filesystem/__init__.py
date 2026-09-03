"""Filesystem, caching and video helpers.

``get_data_directory``, ``compute_with_cache_check`` and ``mp4_to_gif`` are
pure-stdlib/numpy and always available. ``video_from_image_dir`` needs Pillow
(``pip install 'simkit[video]'``); if Pillow is missing the name is simply not
exported, so importing this package still succeeds on a lean install.
"""

from .get_data_directory import get_data_directory
from .compute_with_cache_check import compute_with_cache_check

# ffmpeg is invoked as a subprocess -- no Python dependency beyond stdlib.
from .mp4_to_gif import mp4_to_gif

try:  # Pillow-dependent -- pip install 'simkit[video]'
    from .video_from_image_dir import video_from_image_dir
except ImportError:  # pragma: no cover - depends on optional install
    pass
