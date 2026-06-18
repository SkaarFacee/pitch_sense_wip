"""
PitchSense Flask Server — Modern EA FC-style web UI for football match analysis.

Endpoints:
    GET  /                       → Main SPA
    GET  /api/videos             → List available match videos
    GET  /api/models             → Model availability status
    POST /api/process            → Start a video processing job (SSE-friendly)
    GET  /api/progress/<job_id>  → Server-Sent Events stream of per-frame progress
    GET  /api/analytics/<job_id> → Full JSON analytics payload for completed job
    GET  /api/video/<job_id>/<name> → Stream processed output video
    GET  /static/...             → Static assets

Run:
    python -m app.server          (or)   python app/server.py
"""

from __future__ import annotations

import sys
import json
import time
import uuid
import threading
import re
import queue
from collections import Counter
from pathlib import Path
from typing import Optional

# Make package importable from project root
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    abort,
)

from constants import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    CENTER_X,
    CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_WIDTH,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    PENALTY_ARC_RADIUS,
    LEFT_PENALTY_X,
    RIGHT_PENALTY_X,
    LEFT_GOAL_AREA_X,
    RIGHT_GOAL_AREA_X,
    PENALTY_Y_TOP,
    PENALTY_Y_BOTTOM,
    GOAL_AREA_Y_TOP,
    GOAL_AREA_Y_BOTTOM,
    LEFT_PENALTY_SPOT_X,
    RIGHT_PENALTY_SPOT_X,
)
from keypoint_pipeline import KeypointPipeline
from game_analyzer import GameAnalyzer


# ─── Configuration ────────────────────────────────────────────────────────────
TEST_DATA_DIR = _PROJECT_ROOT / "data" / "matches"
OUTPUT_BASE = _PROJECT_ROOT / "output"

MODEL_PATHS = {
    "keypoint": str(_PROJECT_ROOT / "models" / "keypoint_model" / "26n_pipeline" / "no_aug" / "weights" / "best.pt"),
    "player": str(_PROJECT_ROOT / "models" / "player_model" / "best.pt"),
    "seg": str(_PROJECT_ROOT / "models" / "segmentation" / "best.pt"),
    "ball": str(_PROJECT_ROOT / "models" / "ball_model" / "yolo26_best.pt"),
}

SUPPORTED_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov", ".mkv")

OUTPUT_VIDEOS = [
    ("final_draft.mp4", "Final Draft (Main + Pitch PIP)"),
    ("annotated_video.mp4", "Annotated (Keypoints + Team Bboxes + Ball)"),
    ("deep_analysis.mp4", "Deep Analysis (Segmentation Overlay + Ball)"),
    ("full_pitch_debug_map.mp4", "Full Pitch Map (Top-Down View + Ball Trail)"),
    ("keypoint_annotations.mp4", "Keypoint Annotations (Keypoints on Original)"),
]

SEG_CLASS_LABELS = {
    "18Yard": "Penalty Area (18yd)",
    "18Yard Circle": "Penalty Arc",
    "5Yard": "Goal Area (6yd)",
    "Half Central Circle": "Center Circle",
    "Half Field": "Half Field",
}


# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)
app.config["JSON_SORT_KEYS"] = False


# ─── Job Registry (in-memory) ─────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _new_job_state(video_path: str, total_frames: int, output_dir: Path) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "video_path": video_path,
        "video_name": Path(video_path).stem,
        "output_dir": str(output_dir),
        "status": "queued",  # queued | running | done | error
        "error": None,
        "total_frames": total_frames,
        "processed": 0,
        "ball_detected": 0,
        "started_at": None,
        "finished_at": None,
        "events": queue.Queue(),   # SSE message bus
        "analytics_data": [],
        "game_data": [],
    }


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name)


def _remove_emojis(text: str) -> str:
    return re.sub(r"[^\w\s.\-]", "", text).strip()


def _get_video_files() -> list[Path]:
    files: list[Path] = []
    if not TEST_DATA_DIR.exists():
        return files
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(TEST_DATA_DIR.glob(f"*{ext}"))
    return sorted(files)


def _get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(total, 0)


def _check_models() -> dict:
    return {name: Path(path).exists() for name, path in MODEL_PATHS.items()}


# ─── Analytics helpers ────────────────────────────────────────────────────────
def _build_seg_analytics(analytics_data: list) -> dict:
    class_counter: Counter = Counter()
    per_frame: dict[int, list[str]] = {}
    frames_with_seg = 0
    for entry in analytics_data:
        frame_idx = entry["frame_idx"]
        segments = entry.get("segments", [])
        if segments:
            frames_with_seg += 1
        frame_classes = []
        for seg in segments:
            cn = seg.get("class_name", "unknown")
            frame_classes.append(cn)
            class_counter[cn] += 1
        per_frame[frame_idx] = frame_classes

    classes_sorted = sorted(class_counter.items(), key=lambda kv: kv[1], reverse=True)
    total_detections = sum(class_counter.values())

    return {
        "class_frequency": [
            {
                "key": k,
                "label": SEG_CLASS_LABELS.get(k, k),
                "count": v,
                "pct": round((v / total_detections) * 100, 1) if total_detections else 0.0,
            }
            for k, v in classes_sorted
        ],
        "frames_with_seg": frames_with_seg,
        "total_frames": len(analytics_data),
        "total_detections": total_detections,
        "coverage_pct": round((frames_with_seg / max(len(analytics_data), 1)) * 100, 1),
    }


def _pitch_outline_geometry() -> dict:
    """Return geometry (lines, circles, arcs) of a standard pitch — for Plotly rendering."""
    # Lines as lists of (x1, y1, x2, y2)
    lines = [
        # outer boundary
        (0, 0, PITCH_LENGTH, 0),
        (PITCH_LENGTH, 0, PITCH_LENGTH, PITCH_WIDTH),
        (PITCH_LENGTH, PITCH_WIDTH, 0, PITCH_WIDTH),
        (0, PITCH_WIDTH, 0, 0),
        # midline
        (CENTER_X, 0, CENTER_X, PITCH_WIDTH),
        # left penalty
        (0, PENALTY_Y_TOP, LEFT_PENALTY_X, PENALTY_Y_TOP),
        (LEFT_PENALTY_X, PENALTY_Y_TOP, LEFT_PENALTY_X, PENALTY_Y_BOTTOM),
        (LEFT_PENALTY_X, PENALTY_Y_BOTTOM, 0, PENALTY_Y_BOTTOM),
        # right penalty
        (PITCH_LENGTH, PENALTY_Y_TOP, RIGHT_PENALTY_X, PENALTY_Y_TOP),
        (RIGHT_PENALTY_X, PENALTY_Y_TOP, RIGHT_PENALTY_X, PENALTY_Y_BOTTOM),
        (RIGHT_PENALTY_X, PENALTY_Y_BOTTOM, PITCH_LENGTH, PENALTY_Y_BOTTOM),
        # left 6-yard
        (0, GOAL_AREA_Y_TOP, LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP),
        (LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP, LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM),
        (LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM, 0, GOAL_AREA_Y_BOTTOM),
        # right 6-yard
        (PITCH_LENGTH, GOAL_AREA_Y_TOP, RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP),
        (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP, RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM),
        (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM, PITCH_LENGTH, GOAL_AREA_Y_BOTTOM),
    ]
    # Center circle as polyline
    theta = np.linspace(0, 2 * np.pi, 64)
    cc_x = (CENTER_X + CENTER_CIRCLE_RADIUS * np.cos(theta)).tolist()
    cc_y = (CENTER_Y + CENTER_CIRCLE_RADIUS * np.sin(theta)).tolist()

    # Penalty arcs
    th = np.arccos((LEFT_PENALTY_X - LEFT_PENALTY_SPOT_X) / PENALTY_ARC_RADIUS)
    ang_left = np.linspace(-th, th, 32)
    ang_right = np.linspace(np.pi - th, np.pi + th, 32)
    arc_l_x = (LEFT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS * np.cos(ang_left)).tolist()
    arc_l_y = (CENTER_Y + PENALTY_ARC_RADIUS * np.sin(ang_left)).tolist()
    arc_r_x = (RIGHT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS * np.cos(ang_right)).tolist()
    arc_r_y = (CENTER_Y + PENALTY_ARC_RADIUS * np.sin(ang_right)).tolist()

    return {
        "length": PITCH_LENGTH,
        "width": PITCH_WIDTH,
        "lines": lines,
        "center_circle": {"x": cc_x, "y": cc_y},
        "center_spot": {"x": CENTER_X, "y": CENTER_Y},
        "penalty_spot_left": {"x": LEFT_PENALTY_SPOT_X, "y": CENTER_Y},
        "penalty_spot_right": {"x": RIGHT_PENALTY_SPOT_X, "y": CENTER_Y},
        "arc_left": {"x": arc_l_x, "y": arc_l_y},
        "arc_right": {"x": arc_r_x, "y": arc_r_y},
    }


def _heatmap_payload(game_data: list, bins=(18, 12)) -> dict:
    """Compute a heatmap grid + sample positions for Plotly hover tooltips."""
    heat = GameAnalyzer.compute_heatmaps(game_data, bins=bins)
    x_edges: np.ndarray = heat["x_edges"]
    y_edges: np.ndarray = heat["y_edges"]
    # Cell centers for Plotly heatmap rendering
    x_centers = ((x_edges[:-1] + x_edges[1:]) / 2.0).tolist()
    y_centers = ((y_edges[:-1] + y_edges[1:]) / 2.0).tolist()

    def _zone_label(xc: float) -> str:
        if xc < PITCH_LENGTH / 3.0:
            return "Defensive Third"
        if xc < 2.0 * PITCH_LENGTH / 3.0:
            return "Middle Third"
        return "Attacking Third"

    def _grid(matrix: np.ndarray):
        # matrix shape: (n_x, n_y) → Plotly expects z[row=y][col=x]
        z = matrix.T.tolist()  # rows = y, cols = x
        labels = [
            [_zone_label(xc) for xc in x_centers]
            for _ in y_centers
        ]
        return {"z": z, "x_centers": x_centers, "y_centers": y_centers, "zone_labels": labels}

    return {
        "bins": list(bins),
        "team1": _grid(heat["team1_heatmap"]),
        "team2": _grid(heat["team2_heatmap"]),
        "team1_count": int(heat["team1_count"]),
        "team2_count": int(heat["team2_count"]),
    }


def _formation_scatter_payload(game_data: list, max_frames: int = 200) -> dict:
    from game_analyzer import TEAM0, TEAM1
    registry = GameAnalyzer.build_registry(game_data)
    step = max(1, len(game_data) // max_frames)
    t1_xs, t1_ys, t1_frames = [], [], []
    t2_xs, t2_ys, t2_frames = [], [], []
    for entry in game_data[::step]:
        tids = entry.get("track_ids")
        positions = entry.get("player_positions")
        fi = entry.get("frame_idx", 0)
        if registry.has_track_ids and tids is not None and positions is not None and len(tids) == len(positions):
            for i, tid in enumerate(np.asarray(tids)):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                if rec.canonical_team not in (TEAM0, TEAM1):
                    continue
                x = float(positions[i][0]); y = float(positions[i][1])
                if not (-2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2):
                    continue
                if rec.canonical_team == TEAM0:
                    t1_xs.append(x); t1_ys.append(y); t1_frames.append(fi)
                else:
                    t2_xs.append(x); t2_ys.append(y); t2_frames.append(fi)
        else:
            _, _, t1, t2 = GameAnalyzer._split_teams(entry)
            if t1 is not None and len(t1) > 0:
                for x, y in t1:
                    if -2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2:
                        t1_xs.append(float(x))
                        t1_ys.append(float(y))
                        t1_frames.append(fi)
            if t2 is not None and len(t2) > 0:
                for x, y in t2:
                    if -2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2:
                        t2_xs.append(float(x))
                        t2_ys.append(float(y))
                        t2_frames.append(fi)
    return {
        "team1": {"x": t1_xs, "y": t1_ys, "frame": t1_frames},
        "team2": {"x": t2_xs, "y": t2_ys, "frame": t2_frames},
    }


def _ball_trail_payload(game_data: list) -> dict:
    xs, ys, frames = [], [], []
    for entry in game_data:
        bp = entry.get("ball_position")
        if bp is None:
            continue
        bp = np.asarray(bp, dtype=float).reshape(-1)
        if bp.shape[0] < 2:
            continue
        x, y = float(bp[0]), float(bp[1])
        if -5 <= x <= PITCH_LENGTH + 5 and -5 <= y <= PITCH_WIDTH + 5:
            xs.append(x)
            ys.append(y)
            frames.append(int(entry.get("frame_idx", 0)))
    return {"x": xs, "y": ys, "frame": frames}


def _possession_timeline(game_data: list, window: int = 30) -> dict:
    """Rolling possession % over time (window in frames) for line chart.

    Track-aware: uses the canonical per-track team from the registry, so a
    single-frame misclassification of one player cannot flip the
    possession assignment for that frame.
    """
    registry = GameAnalyzer.build_registry(game_data)
    t1_rolling = []
    t2_rolling = []
    frames_axis = []
    buf = []  # FIFO of 0/1/-1 (team in possession of nearest player to ball)

    for entry in game_data:
        ball = entry.get("ball_position")
        team = -1
        if ball is not None:
            ball_arr = np.asarray(ball, dtype=np.float32).reshape(1, 2)
            winner = GameAnalyzer._nearest_team_to_ball(entry, ball_arr, registry)
            if winner is not None:
                team = int(winner)
        buf.append(team)
        if len(buf) > window:
            buf.pop(0)
        valid = [b for b in buf if b in (0, 1)]
        if valid:
            t1pc = round(100.0 * sum(1 for b in valid if b == 0) / len(valid), 1)
            t2pc = round(100.0 * sum(1 for b in valid if b == 1) / len(valid), 1)
        else:
            t1pc = 0.0
            t2pc = 0.0
        t1_rolling.append(t1pc)
        t2_rolling.append(t2pc)
        frames_axis.append(int(entry.get("frame_idx", len(frames_axis) + 1)))
    return {"frame": frames_axis, "team1": t1_rolling, "team2": t2_rolling, "window": window}


def _team_radar_payload(formation: dict, stats: dict, possession: dict) -> dict:
    """EA-FC-style radar comparing the two teams on 6 axes (0–100)."""
    # Normalise / scale each metric to 0–100 sensible range
    def clip(v, lo=0.0, hi=100.0):
        return float(max(lo, min(hi, v)))

    t1_attack = clip(formation["team1_avg_center"][0] / PITCH_LENGTH * 100) if formation.get("team1_avg_center") else 50.0
    t2_attack = clip(formation["team2_avg_center"][0] / PITCH_LENGTH * 100) if formation.get("team2_avg_center") else 50.0
    # Defensive depth — lower min X = lower number, so flip
    t1_def = clip(100 - (formation["team1_defensive_depth"] / PITCH_LENGTH * 100))
    t2_def = clip(100 - (formation["team2_defensive_depth"] / PITCH_LENGTH * 100))
    # Compactness = inverse of spread (smaller = more compact)
    max_spread = 40.0
    t1_comp = clip(100 - (formation["team1_avg_spread"] / max_spread * 100))
    t2_comp = clip(100 - (formation["team2_avg_spread"] / max_spread * 100))
    # Width = spread again but interpreted as field coverage
    t1_width = clip(formation["team1_avg_spread"] / max_spread * 100)
    t2_width = clip(formation["team2_avg_spread"] / max_spread * 100)
    # Possession
    t1_poss = clip(possession.get("team1_possession_pct", 0.0))
    t2_poss = clip(possession.get("team2_possession_pct", 0.0))
    # Tempo proxy = ball progression scaled
    tempo = clip(stats.get("ball_progression_m", 0) / 1000.0 * 100)
    return {
        "categories": ["Attacking Intent", "Defensive Depth", "Compactness", "Width", "Possession", "Tempo"],
        "team1": [t1_attack, t1_def, t1_comp, t1_width, t1_poss, tempo],
        "team2": [t2_attack, t2_def, t2_comp, t2_width, t2_poss, tempo],
    }


def _build_full_analytics(job: dict) -> dict:
    game_data = job["game_data"]
    analytics_data = job["analytics_data"]

    if not game_data:
        return {
            "ok": False,
            "error": "No game data was produced — was the video processed?",
        }

    possession = GameAnalyzer.compute_possession(game_data, "Team 1", "Team 2")
    formation = GameAnalyzer.compute_formation(game_data)
    territory = GameAnalyzer.compute_territory(game_data)
    stats = GameAnalyzer.compute_match_stats(game_data)
    seg_summary = _build_seg_analytics(analytics_data)

    # JSON-safe enrichment
    formation_json = {
        "team1_avg_center": formation["team1_avg_center"],
        "team2_avg_center": formation["team2_avg_center"],
        "team1_avg_spread": formation["team1_avg_spread"],
        "team2_avg_spread": formation["team2_avg_spread"],
        "team1_defensive_depth": formation["team1_defensive_depth"],
        "team2_defensive_depth": formation["team2_defensive_depth"],
        "frames_with_players": formation["frames_with_players"],
    }

    payload = {
        "ok": True,
        "job_id": job["id"],
        "video_name": job["video_name"],
        "pitch": _pitch_outline_geometry(),
        "possession": possession,
        "possession_timeline": _possession_timeline(game_data),
        "formation": formation_json,
        "formation_scatter": _formation_scatter_payload(game_data),
        "heatmaps": _heatmap_payload(game_data),
        "ball_trail": _ball_trail_payload(game_data),
        "territory": territory,
        "stats": stats,
        "segments": seg_summary,
        "radar": _team_radar_payload(formation_json, stats, possession),
        "team_colors": {
            "team1_bgr": list(GameAnalyzer.dominant_team_bgr(game_data, team=0))
                          if GameAnalyzer.dominant_team_bgr(game_data, team=0) else None,
            "team2_bgr": list(GameAnalyzer.dominant_team_bgr(game_data, team=1))
                          if GameAnalyzer.dominant_team_bgr(game_data, team=1) else None,
        },
        "output_videos": [
            {"file": fn, "label": lab}
            for fn, lab in OUTPUT_VIDEOS
            if (Path(job["output_dir"]) / fn).exists()
        ],
    }
    return payload


# ─── Background processing thread ─────────────────────────────────────────────
def _process_job(job_id: str, max_frames: Optional[int], enable_team_colors: bool,
                 flip_projection_x: bool = False, flip_projection_y: bool = True):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return

    def push(event_type: str, payload: dict):
        try:
            job["events"].put_nowait({"type": event_type, "data": payload})
        except Exception:
            pass

    try:
        job["status"] = "running"
        job["started_at"] = time.time()
        push("status", {"status": "running", "message": "Initialising pipeline…"})

        pipeline = KeypointPipeline(
            keypoint_model_path=MODEL_PATHS["keypoint"],
            player_model_path=MODEL_PATHS["player"],
            seg_model_path=MODEL_PATHS["seg"],
            ball_model_path=MODEL_PATHS["ball"],
            enable_team_colors=enable_team_colors,
            flip_projection_x=flip_projection_x,
            flip_projection_y=flip_projection_y,
        )

        total = job["total_frames"]
        if max_frames is not None and max_frames > 0:
            total = min(total, max_frames)

        processed = 0
        ball_count = 0

        for result in pipeline.process_video(
            source_video_path=job["video_path"],
            output_dir=job["output_dir"],
            start_frame=0,
            max_frames=max_frames if max_frames and max_frames > 0 else None,
        ):
            processed += 1
            has_ball = len(result.get("ball_xyxy", [])) > 0
            if has_ball:
                ball_count += 1

            segs = result.get("processed_segments", [])
            if segs:
                # Strip non-serialisable parts before storing
                job["analytics_data"].append({
                    "frame_idx": processed,
                    "segments": [
                        {
                            "class_name": s.get("class_name"),
                            "confidence": float(s.get("confidence", 0.0)),
                        }
                        for s in segs
                    ],
                })

            team_info = result.get("team_info")
            team_ids = team_info.get("team_ids") if team_info else None
            track_ids = result.get("track_ids", np.empty((0,), dtype=np.int32))
            track_quality = team_info.get("track_quality") if team_info else None
            team1_bgr = team_info.get("team1_bgr") if team_info else None
            team2_bgr = team_info.get("team2_bgr") if team_info else None
            job["game_data"].append({
                "frame_idx": processed,
                "player_positions": result.get("player_pitch_pts", np.empty((0, 2))),
                "team_ids": team_ids,
                "track_ids": track_ids,
                "track_quality": track_quality,
                "team1_bgr": team1_bgr,
                "team2_bgr": team2_bgr,
                "ball_position": result.get("ball_pitch_pt"),
                "player_conf": result.get("player_conf", np.empty((0,))),
            })

            job["processed"] = processed
            job["ball_detected"] = ball_count

            pct = (processed / max(total, 1)) * 100.0
            h_mode = result.get("H_info", {}).get("mode", "N/A")
            n_players = int(len(result.get("player_pitch_pts", [])))
            push("progress", {
                "processed": processed,
                "total": total,
                "pct": round(pct, 2),
                "ball_detected_total": ball_count,
                "has_ball": bool(has_ball),
                "players": n_players,
                "homography": str(h_mode),
            })

        job["status"] = "done"
        job["finished_at"] = time.time()
        push("done", {"processed": processed, "ball_detected": ball_count})

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        push("error", {"message": str(e)})


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos")
def api_videos():
    files = _get_video_files()
    return jsonify({
        "videos": [
            {
                "name": _remove_emojis(f.stem),
                "path": str(f),
                "filename": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1) if f.exists() else 0,
            }
            for f in files
        ],
        "data_dir": str(TEST_DATA_DIR),
    })


@app.route("/api/models")
def api_models():
    status = _check_models()
    return jsonify({
        "models": [
            {"name": name, "path": MODEL_PATHS[name], "ready": ready}
            for name, ready in status.items()
        ],
        "all_ready": all(status.values()),
    })


@app.route("/api/process", methods=["POST"])
def api_process():
    payload = request.get_json(silent=True) or {}
    video_path = payload.get("video_path")
    max_frames = payload.get("max_frames", 0)
    enable_team_colors = bool(payload.get("enable_team_colors", True))
    flip_projection_x = bool(payload.get("flip_projection_x", False))
    flip_projection_y = bool(payload.get("flip_projection_y", True))

    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400
    if not all(_check_models().values()):
        return jsonify({"ok": False, "error": "One or more models are missing"}), 400

    total = _get_total_frames(video_path)
    out_dir = OUTPUT_BASE / f"processed_{_safe_stem(Path(video_path).stem)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    job = _new_job_state(video_path, total, out_dir)
    with JOBS_LOCK:
        JOBS[job["id"]] = job

    t = threading.Thread(
        target=_process_job,
        args=(job["id"], int(max_frames) if max_frames else None,
              enable_team_colors, flip_projection_x, flip_projection_y),
        daemon=True,
    )
    t.start()

    return jsonify({
        "ok": True,
        "job_id": job["id"],
        "total_frames": total,
        "output_dir": str(out_dir),
    })


@app.route("/api/progress/<job_id>")
def api_progress(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job"}), 404

    def stream():
        # Initial snapshot
        yield "event: snapshot\ndata: " + json.dumps({
            "status": job["status"],
            "processed": job["processed"],
            "total": job["total_frames"],
            "ball_detected_total": job["ball_detected"],
        }) + "\n\n"

        # Drain queue
        while True:
            try:
                msg = job["events"].get(timeout=30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"event: {msg['type']}\ndata: {json.dumps(msg['data'])}\n\n"
            if msg["type"] in {"done", "error"}:
                break

    return Response(stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/api/analytics/<job_id>")
def api_analytics(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"ok": False, "error": f"Job status: {job['status']}"}), 409
    return jsonify(_build_full_analytics(job))


@app.route("/api/video/<job_id>/<name>")
def api_video(job_id: str, name: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)
    path = Path(job["output_dir"]) / name
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="video/mp4", conditional=True)


# ─── Entrypoint ───────────────────────────────────────────────────────────────
def main():
    host = "127.0.0.1"
    port = 5050
    print(f"\n  ⚽  PitchSense server starting on  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
