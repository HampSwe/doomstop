"""Plug-in point for teammates' classifiers.

Replace the bodies of these three functions with the real models. The
recorder calls them and handles everything else (camera ownership, timing,
rolling windows, logging). Return None when you have no opinion for a frame
or block (e.g. no face found); it is logged as missing, not as False.

Camera: the recorder opens the camera once and passes every frame to BOTH
head_is_down and phone_in_frame, so the two models never fight over the
device. `frame` is a BGR numpy array (H, W, 3), as OpenCV gives it.
"""

from __future__ import annotations

import numpy as np


def head_is_down(frame: np.ndarray) -> bool | None:
    """Logitech head-pose model: True if the user is looking down."""
    return None  # TODO(teammate): plug in the look-down classifier


def phone_in_frame(frame: np.ndarray) -> bool | None:
    """Object detector: True if a phone is visible in the frame."""
    return None  # TODO(teammate): plug in the phone detector


def audio_reel_score(samples: np.ndarray, sample_rate: int) -> float | None:
    """Optional audio model: probability-like score that a reel is playing.

    `samples` is a mono float32 block (~0.25 s). Hand-made audio features
    (loudness, spectral centroid, zero-crossing rate) are logged anyway, so
    leaving this as None is fine.
    """
    return None  # TODO(teammate): plug in the reel-audio model
