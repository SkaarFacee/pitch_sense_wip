"""demo_loader.py — load pre-cached demo matches from `demo_cache/`.

A demo bundle is a directory under the repo-root ``demo_cache/`` folder
populated by ``scripts/setup_demo_cache.py``. Each bundle carries:

    <demo_id>/
        manifest.json          # title, source video, fps, total_frames, file list
        meta.json              # summary KPIs (possession, ball detection %, ...)
        videos/*.mp4           # the 5 output videos
        data/game_data.npz     # per-frame analytics, written by KeypointPipeline
        data/analytics_data.json

Loading returns a ``dict`` shaped exactly like the live Streamlit loop
produces (``game_data`` + ``analytics_data`` + ``last_output_dir`` +
``fps`` + ``total_frames``) so the existing tabs can render the demo
without any branching.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
DEMO_CACHE_DIR = _PROJECT_ROOT / "demo_cache"

# Sentinel values written by KeypointPipeline._persist_per_frame_data when
# a per-frame field was None in the original entry. The loader checks for
# these to know whether to emit None or the real value.
_NONE_FRAME_SENTINEL = np.asarray([-1], dtype=np.int32)
_NONE_BGR_SENTINEL = np.asarray([-1, -1, -1], dtype=np.int32)


class DemoLoadError(RuntimeError):
    """Raised when a demo cannot be loaded (missing files, corrupt data, ...).

    The message is written for end users — Streamlit surfaces it as a
    non-fatal warning in the demo picker.
    """


@dataclass
class DemoInfo:
    id: str
    title: str
    source_video: str
    fps: float
    total_frames: int
    cache_dir: Path
    files: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def is_complete(self) -> bool:
        """True iff every declared file in ``files`` exists on disk."""
        if not self.files:
            return False
        for rel in self.files:
            if not (self.cache_dir / rel).exists():
                return False
        return True


# ─── Discovery ────────────────────────────────────────────────────────────────
def list_demos() -> list[DemoInfo]:
    """Return every demo declared under ``DEMO_CACHE_DIR``.

    Reads the top-level ``manifest.json`` if present; otherwise scans for
    per-demo ``manifest.json`` files in immediate subdirectories. Demos
    with no manifest are silently skipped (they're considered partial /
    in-progress bundles from a half-completed setup).
    """
    if not DEMO_CACHE_DIR.exists():
        return []

    top_manifest = DEMO_CACHE_DIR / "manifest.json"
    demos: list[DemoInfo] = []

    if top_manifest.exists():
        try:
            data = json.loads(top_manifest.read_text())
        except json.JSONDecodeError:
            data = {}
        for entry in (data.get("demos") or []):
            info = _entry_to_info(entry)
            if info is not None:
                demos.append(info)

    # Backfill: also include any per-demo manifests the top-level manifest
    # didn't mention (e.g. a fresh bundle added but top-level manifest not
    # regenerated yet). Idempotent: de-duplicates by ``id``.
    seen = {d.id for d in demos}
    for sub in sorted(DEMO_CACHE_DIR.iterdir()):
        if not sub.is_dir():
            continue
        per = sub / "manifest.json"
        if not per.exists():
            continue
        try:
            data = json.loads(per.read_text())
        except json.JSONDecodeError:
            continue
        info = _entry_to_info(data, cache_dir=sub)
        if info is not None and info.id not in seen:
            demos.append(info)
            seen.add(info.id)

    return sorted(demos, key=lambda d: d.title.lower())


def _entry_to_info(entry: dict, cache_dir: Optional[Path] = None) -> Optional[DemoInfo]:
    if not isinstance(entry, dict):
        return None
    demo_id = entry.get("id")
    if not demo_id:
        return None
    if cache_dir is None:
        cache_dir = DEMO_CACHE_DIR / demo_id
    meta_path = cache_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    files = list(entry.get("files") or [])
    return DemoInfo(
        id=demo_id,
        title=str(entry.get("title") or demo_id),
        source_video=str(entry.get("source_video") or ""),
        fps=float(entry.get("fps") or meta.get("fps") or 30.0),
        total_frames=int(entry.get("total_frames") or meta.get("total_frames") or 0),
        cache_dir=cache_dir,
        files=files,
        meta=meta,
    )


# ─── Validation ───────────────────────────────────────────────────────────────
def validate_demo(demo_id: str) -> list[str]:
    """Return a list of human-readable warnings for a demo's missing files.

    Empty list means every declared file is on disk. The Streamlit picker
    uses this to show non-fatal warnings instead of failing silently.
    """
    info = _find_demo(demo_id)
    if info is None:
        return [f"Demo '{demo_id}' not found in {DEMO_CACHE_DIR}"]
    warnings: list[str] = []
    if not info.files:
        warnings.append("manifest.json has no 'files' entry — bundle may be incomplete")
    for rel in info.files:
        if not (info.cache_dir / rel).exists():
            warnings.append(f"missing: {rel}")
    # Required structural files even if not listed.
    for rel in ("videos/final_draft.mp4", "data/game_data.npz"):
        if not (info.cache_dir / rel).exists() and rel not in info.files:
            warnings.append(f"missing (required): {rel}")
    return warnings


def _find_demo(demo_id: str) -> Optional[DemoInfo]:
    for d in list_demos():
        if d.id == demo_id:
            return d
    return None


# ─── Loading ──────────────────────────────────────────────────────────────────
def load_demo(demo_id: str) -> dict:
    """Load a demo bundle and return a dict shaped for the Streamlit app.

    Returned keys::

        {
            "game_data":     [ {frame_idx, player_positions, ...}, ... ],
            "analytics_data":[ {frame_idx, segments}, ... ],
            "last_output_dir": "<demo_cache>/<id>/videos",
            "fps":           float,
            "total_frames":  int,
            "meta":          {...},   # raw meta.json contents
            "info":          DemoInfo,
        }

    Raises ``DemoLoadError`` on any failure with a user-facing message.
    """
    info = _find_demo(demo_id)
    if info is None:
        raise DemoLoadError(f"Demo '{demo_id}' not found in {DEMO_CACHE_DIR}")
    warnings = validate_demo(demo_id)
    hard_missing = [w for w in warnings if "missing (required)" in w or "not found" in w]
    if hard_missing:
        raise DemoLoadError(
            f"Demo '{demo_id}' is incomplete: " + "; ".join(hard_missing)
        )

    game_data = _load_game_data(info.cache_dir / "data" / "game_data.npz")
    analytics_data = _load_analytics_data(info.cache_dir / "data" / "analytics_data.json")
    last_output_dir = str(info.cache_dir / "videos")

    return {
        "game_data": game_data,
        "analytics_data": analytics_data,
        "last_output_dir": last_output_dir,
        "fps": float(info.fps),
        "total_frames": int(info.total_frames),
        "meta": dict(info.meta),
        "info": info,
    }


def _load_analytics_data(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DemoLoadError(f"Corrupt analytics_data.json: {exc}") from exc
    if not isinstance(data, list):
        return []
    return data


def _load_game_data(path: Path) -> list[dict]:
    if not path.exists():
        raise DemoLoadError(f"Missing game_data archive: {path}")
    try:
        with np.load(path, allow_pickle=True) as npz:
            keys = list(npz.files)
            if "frame_idx" not in keys:
                raise DemoLoadError(
                    "game_data.npz is missing the 'frame_idx' column"
                )
            # Pre-fetch every column once into Python-owned arrays so we
            # can iterate over them after the npz file handle closes.
            cols: dict[str, np.ndarray] = {}
            for key in keys:
                cols[key] = np.asarray(npz[key])
    except DemoLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise DemoLoadError(f"Could not open game_data.npz: {exc}") from exc

    n = int(cols["frame_idx"].shape[0])
    game_data: list[dict] = []
    for i in range(n):
        entry = _rebuild_entry(cols, i)
        game_data.append(entry)
    return game_data


def _rebuild_entry(cols: dict[str, np.ndarray], i: int) -> dict:
    """Reconstruct a single per-frame entry from the columnar npz file."""
    frame_idx = _scalar_int(cols["frame_idx"][i], default=0)

    player_positions = _row_2d(cols.get("player_positions"), i, (0, 2), np.float32)
    player_xyxy = _row_2d(cols.get("player_xyxy"), i, (0, 4), np.float32)
    ball_xyxy = _row_2d(cols.get("ball_xyxy"), i, (0, 4), np.float32)

    team_ids = _row_nullable_int(cols.get("team_ids"), i)
    role_ids = _row_nullable_int(cols.get("role_ids"), i)
    identity_ids = _row_nullable_int(cols.get("identity_ids"), i)
    track_ids = _row_nullable_int(cols.get("track_ids"), i)
    track_quality = _row_nullable_floats(cols.get("track_quality"), i)
    player_conf = _row_nullable_floats(cols.get("player_conf"), i)
    ball_conf = _row_nullable_floats(cols.get("ball_conf"), i)

    team1_bgr = _row_bgr(cols.get("team1_bgr"), i)
    team2_bgr = _row_bgr(cols.get("team2_bgr"), i)

    ball_position = _row_ball_position(cols.get("ball_position"), i)
    pass_event = _row_pass_event(cols.get("pass_event"), i)

    return {
        "frame_idx": int(frame_idx),
        "player_positions": player_positions,
        "player_xyxy": player_xyxy,
        "team_ids": team_ids,
        "role_ids": role_ids,
        "identity_ids": identity_ids,
        "track_ids": track_ids,
        "track_quality": track_quality,
        "team1_bgr": team1_bgr,
        "team2_bgr": team2_bgr,
        "ball_position": ball_position,
        "ball_xyxy": ball_xyxy,
        "ball_conf": ball_conf,
        "player_conf": player_conf,
        "pass_event": pass_event,
    }


def _scalar_int(arr, default: int = 0) -> int:
    try:
        return int(np.asarray(arr).reshape(-1)[0])
    except Exception:
        return int(default)


def _row_2d(cols_opt, i: int, empty_shape: tuple[int, int], dtype) -> np.ndarray:
    if cols_opt is None:
        return np.empty(empty_shape, dtype=dtype)
    arr = np.asarray(cols_opt[i])
    if arr.ndim == 2:
        return arr.astype(dtype, copy=False)
    if arr.size == 0:
        return np.empty(empty_shape, dtype=dtype)
    # 1-D fallback (shouldn't happen for 2-D columns but stay defensive).
    n_cols = empty_shape[1]
    n_rows = arr.size // n_cols
    return arr.reshape(n_rows, n_cols).astype(dtype, copy=False) if n_rows else np.empty(empty_shape, dtype=dtype)


def _row_nullable_int(cols_opt, i: int):
    """Return the row's int array, or None when the row is the None sentinel."""
    if cols_opt is None:
        return None
    arr = np.asarray(cols_opt[i])
    if arr.size == _NONE_FRAME_SENTINEL.size and np.array_equal(
        arr.astype(np.int32, copy=False).reshape(-1),
        _NONE_FRAME_SENTINEL.reshape(-1),
    ):
        return None
    if arr.size == 0:
        return np.empty((0,), dtype=np.int32)
    return arr.astype(np.int32, copy=False)


def _row_nullable_floats(cols_opt, i: int):
    if cols_opt is None:
        return np.empty((0,), dtype=np.float32)
    arr = np.asarray(cols_opt[i])
    if arr.size == _NONE_FRAME_SENTINEL.size and np.array_equal(
        arr.astype(np.int32, copy=False).reshape(-1),
        _NONE_FRAME_SENTINEL.reshape(-1),
    ):
        return None
    if arr.size == 0:
        return np.empty((0,), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _row_bgr(cols_opt, i: int):
    """Return a BGR tuple, or None when the row is the None sentinel."""
    if cols_opt is None:
        return None
    arr = np.asarray(cols_opt[i]).reshape(-1)
    if arr.size >= 3 and np.array_equal(arr[:3], _NONE_BGR_SENTINEL):
        return None
    if arr.size < 3:
        return None
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def _row_ball_position(cols_opt, i: int):
    if cols_opt is None:
        return None
    arr = np.asarray(cols_opt[i]).reshape(-1)
    if arr.size < 2:
        return None
    # NaN means missing.
    if not (np.isfinite(arr[0]) and np.isfinite(arr[1])):
        return None
    return arr[:2].astype(np.float32, copy=False)


def _row_pass_event(cols_opt, i: int):
    if cols_opt is None:
        return None
    arr = np.asarray(cols_opt[i], dtype=object)
    if arr.size == 0:
        return None
    val = arr.reshape(-1)[0]
    if val is None:
        return None
    # Numpy may load it as a 0-d object array containing a dict.
    if isinstance(val, dict):
        return val
    return None


# ─── Sanitization ─────────────────────────────────────────────────────────────
def safe_demo_id(stem: str) -> str:
    """Sanitize a video filename stem into a stable demo id.

    Mirrors the existing pipeline's ``"".join(c if c.isalnum() else "_" ...)``
    convention so the same source video maps to the same cache folder
    regardless of which entrypoint created it.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned or "demo"


def manifest_path(demo_id: str) -> Path:
    return DEMO_CACHE_DIR / demo_id / "manifest.json"