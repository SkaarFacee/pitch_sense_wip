"""
frame_scrubber.py — pure helpers for the Frame Inspector scrubber.

No Streamlit imports here so this module is easy to unit-test or reuse
outside the dashboard. All video work is done by reading from disk via
OpenCV; nothing in this module mutates pipeline state.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def get_video_meta(path: str) -> Optional[tuple[float, int, int, int]]:
    """Return (fps, total_frames, width, height) for a video file.

    Returns None if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (fps, total, w, h)
    finally:
        cap.release()


def seek_to_frame(path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Open a video, seek to ``frame_idx`` (1-indexed), and return the BGR frame.

    Returns None if the file cannot be opened or the seek fails. Out-of-range
    indices are clamped to [1, total_frames].
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            frame_idx = max(1, min(int(frame_idx), total))
        else:
            frame_idx = max(1, int(frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def frame_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB for ``st.image``."""
    if bgr is None:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def regions_for_frame(analytics_data: Optional[list], frame_idx: int) -> list[dict]:
    """Return the list of segment dicts detected at ``frame_idx``.

    `analytics_data` is the same list stashed in session_state by the
    Processing tab; entries look like::

        {"frame_idx": int, "segments": [{"class_name": str, "confidence": float}, ...]}

    Returns an empty list when no entry exists for that frame.
    """
    if not analytics_data:
        return []
    for entry in analytics_data:
        if int(entry.get("frame_idx", -1)) == int(frame_idx):
            return list(entry.get("segments", []))
    return []


def find_closest_entry(game_data: Optional[list], frame_idx: int) -> Optional[dict]:
    """Return the per-frame entry whose ``frame_idx`` is closest to ``frame_idx``.

    `game_data` entries are 1-indexed in the same way as the slider. Returns
    None if the list is empty.
    """
    if not game_data:
        return None
    best = None
    best_dist = float("inf")
    for entry in game_data:
        fi = int(entry.get("frame_idx", -1))
        d = abs(fi - int(frame_idx))
        if d < best_dist:
            best_dist = d
            best = entry
    return best
