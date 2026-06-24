"""End-to-end smoke test for the demo cache feature.

Builds a synthetic processed run in output/processed_synthetic/ with a
hand-crafted game_data.npz + analytics_data.json, runs
scripts/setup_demo_cache.py, then verifies demo_loader.load_demo()
returns equivalent data.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from demo_loader import DEMO_CACHE_DIR, list_demos, load_demo, validate_demo


def main() -> int:
    # ─── 1. Build a synthetic processed run ───────────────────────────────
    run_dir = ROOT / "output" / "processed_synthetic_demo_test"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Touch the 5 MP4s so setup_demo_cache sees the run as complete.
    for name in (
        "final_draft.mp4", "deep_analysis.mp4", "full_pitch_debug_map.mp4",
        "annotated_video.mp4", "keypoint_annotations.mp4",
    ):
        (run_dir / name).write_bytes(b"\x00")

    # Synthetic game_data: 5 frames, 3 players on team0, 2 on team1,
    # ball present on frames 0, 2, 4. Mix of missing/non-missing fields.
    n_frames = 5
    frame_idx = np.arange(1, n_frames + 1, dtype=np.int32)

    def _scalar_or_none(values):
        col = []
        for v in values:
            if v is None:
                col.append(np.asarray([-1], dtype=np.int32))
            else:
                col.append(np.asarray(v, dtype=np.int32))
        return np.asarray(col, dtype=object)

    def _2d_or_empty(values, shape):
        col = []
        for v in values:
            if v is None:
                col.append(np.empty(shape, dtype=np.float32))
            else:
                col.append(np.asarray(v, dtype=np.float32))
        return np.asarray(col, dtype=object)

    # Frame 0: 3 players (team0 x2, team1 x1), ball at (52.5, 34)
    # Frame 1: 2 players (team0 x1, team1 x1), no ball
    # Frame 2: 4 players (team0 x2, team1 x2), ball at (90, 34)
    # Frame 3: no players, no ball
    # Frame 4: 3 players (team0 x1, team1 x2), ball at (10, 34), with pass_event

    player_positions = _2d_or_empty([
        np.array([[20.0, 30.0], [30.0, 30.0], [40.0, 30.0]]),  # frame 1
        np.array([[25.0, 30.0], [75.0, 30.0]]),                 # frame 2
        np.array([[15.0, 30.0], [25.0, 30.0], [80.0, 30.0], [90.0, 30.0]]),  # frame 3
        np.empty((0, 2), dtype=np.float32),                     # frame 4
        np.array([[10.0, 30.0], [85.0, 30.0], [95.0, 30.0]]),  # frame 5
    ], (0, 2))

    player_xyxy = _2d_or_empty([
        np.array([[10, 100, 30, 200], [30, 100, 50, 200], [50, 100, 70, 200]]),
        np.array([[25, 100, 45, 200], [75, 100, 95, 200]]),
        np.array([[10, 100, 30, 200], [20, 100, 40, 200], [80, 100, 100, 200], [90, 100, 110, 200]]),
        np.empty((0, 4), dtype=np.float32),
        np.array([[5, 100, 25, 200], [80, 100, 100, 200], [90, 100, 110, 200]]),
    ], (0, 4))

    ball_xyxy = _2d_or_empty([
        np.array([[100, 100, 110, 110]]),
        np.empty((0, 4), dtype=np.float32),
        np.array([[150, 100, 160, 110]]),
        np.empty((0, 4), dtype=np.float32),
        np.array([[50, 100, 60, 110]]),
    ], (0, 4))

    team_ids = _scalar_or_none([
        np.array([0, 0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 0, 1, 1], dtype=np.int32),
        None,
        np.array([0, 1, 1], dtype=np.int32),
    ])
    role_ids = _scalar_or_none([
        np.array([0, 0, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([0, 0, 0, 0], dtype=np.int32),
        None,
        np.array([0, 0, 0], dtype=np.int32),
    ])
    identity_ids = _scalar_or_none([
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        None,
        np.array([1, 3, 4], dtype=np.int32),
    ])
    track_ids = _scalar_or_none([
        np.array([11, 12, 21], dtype=np.int32),
        np.array([11, 21], dtype=np.int32),
        np.array([11, 12, 21, 22], dtype=np.int32),
        None,
        np.array([11, 21, 22], dtype=np.int32),
    ])
    player_conf = _scalar_or_none([
        np.array([0.9, 0.85, 0.8], dtype=np.float32),
        np.array([0.9, 0.85], dtype=np.float32),
        np.array([0.9, 0.85, 0.8, 0.75], dtype=np.float32),
        None,
        np.array([0.9, 0.85, 0.8], dtype=np.float32),
    ])
    ball_conf = _scalar_or_none([
        np.array([0.7], dtype=np.float32),
        None,
        np.array([0.6], dtype=np.float32),
        None,
        np.array([0.65], dtype=np.float32),
    ])
    track_quality = _scalar_or_none([
        np.array([0.9, 0.85, 0.8], dtype=np.float32),
        np.array([0.9, 0.85], dtype=np.float32),
        np.array([0.9, 0.85, 0.8, 0.75], dtype=np.float32),
        None,
        np.array([0.9, 0.85, 0.8], dtype=np.float32),
    ])
    team1_bgr = np.asarray([[0, 0, 200]] * 5, dtype=np.int32)
    team2_bgr = np.asarray([[200, 0, 0]] * 5, dtype=np.int32)
    ball_position = np.asarray([
        [52.5, 34.0],
        [np.nan, np.nan],
        [90.0, 34.0],
        [np.nan, np.nan],
        [10.0, 34.0],
    ], dtype=np.float32)
    pass_event = np.asarray([
        None,
        None,
        {"from_tid": 11, "to_tid": 21, "team": 0, "distance_m": 5.0},
        None,
        None,
    ], dtype=object)

    np.savez_compressed(
        data_dir / "game_data.npz",
        frame_idx=frame_idx,
        player_positions=player_positions,
        player_xyxy=player_xyxy,
        ball_xyxy=ball_xyxy,
        team_ids=team_ids,
        role_ids=role_ids,
        identity_ids=identity_ids,
        track_ids=track_ids,
        track_quality=track_quality,
        player_conf=player_conf,
        ball_conf=ball_conf,
        team1_bgr=team1_bgr,
        team2_bgr=team2_bgr,
        ball_position=ball_position,
        pass_event=pass_event,
    )

    analytics_data = [
        {"frame_idx": 1, "segments": [{"class_name": "Half Field", "confidence": 0.95}]},
        {"frame_idx": 3, "segments": [{"class_name": "18Yard", "confidence": 0.91}]},
    ]
    with open(data_dir / "analytics_data.json", "w") as f:
        json.dump(analytics_data, f)

    # ─── 2. Run setup_demo_cache.py ────────────────────────────────────────
    print("Running setup_demo_cache.py ...")
    result = subprocess.run(
        [sys.executable, "scripts/setup_demo_cache.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    print("stdout:", result.stdout)
    # Exit code 0 = full success; 2 = re-bundle worked but other incomplete
    # runs were flagged. Both are acceptable for this smoke test.
    if result.returncode not in (0, 2):
        print("stderr:", result.stderr)
        return 1

    # ─── 3. Verify the cache contents ──────────────────────────────────────
    demos = list_demos()
    print(f"list_demos() -> {len(demos)} demos:")
    for d in demos:
        print(f"  - {d.id}: {d.title!r} (files={len(d.files)}, complete={d.is_complete()})")

    if not demos:
        print("FAIL: no demos listed after re-bundle")
        return 1

    synthetic = next((d for d in demos if "synthetic_demo_test" in d.id), None)
    if synthetic is None:
        print(f"FAIL: synthetic demo not found in {[d.id for d in demos]}")
        return 1

    warnings = validate_demo(synthetic.id)
    if warnings:
        print(f"WARN: {warnings}")

    # ─── 4. Load the demo and verify round-trip ────────────────────────────
    loaded = load_demo(synthetic.id)
    gd = loaded["game_data"]
    ad = loaded["analytics_data"]
    print(f"loaded: {len(gd)} game_data entries, {len(ad)} analytics entries")
    print(f"fps={loaded['fps']}, total_frames={loaded['total_frames']}")
    print(f"last_output_dir={loaded['last_output_dir']}")

    # Verify shapes
    assert len(gd) == 5, f"expected 5 game_data entries, got {len(gd)}"
    assert gd[0]["frame_idx"] == 1
    assert gd[0]["team_ids"].tolist() == [0, 0, 1]
    assert gd[0]["player_positions"].shape == (3, 2)
    assert gd[0]["ball_position"] is not None
    assert np.allclose(gd[0]["ball_position"], [52.5, 34.0])
    assert gd[1]["ball_position"] is None, f"expected None, got {gd[1]['ball_position']}"
    assert gd[2]["pass_event"] is not None
    assert gd[2]["pass_event"]["from_tid"] == 11
    assert gd[3]["team_ids"] is None
    assert gd[3]["player_positions"].shape == (0, 2)
    assert gd[0]["team1_bgr"] == (0, 0, 200)
    assert gd[0]["team2_bgr"] == (200, 0, 0)

    # Verify analytics
    assert len(ad) == 2
    assert ad[0]["frame_idx"] == 1
    assert ad[0]["segments"][0]["class_name"] == "Half Field"

    # ─── 5. Cleanup ────────────────────────────────────────────────────────
    print("Cleanup ...")
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.rmtree(DEMO_CACHE_DIR, ignore_errors=True)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())