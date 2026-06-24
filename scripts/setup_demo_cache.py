#!/usr/bin/env python3
"""Populate `demo_cache/` with pre-cached output bundles for PitchSense.

Two modes:

    default (re-bundle)
        Scan `output/processed_*` for runs that already have all 5 MP4s and
        a fresh `data/game_data.npz` + `data/analytics_data.json` written
        by `KeypointPipeline` (with `persist_data=True`, which is the
        default since the demo-cache feature landed). Each completed run
        is copied into `demo_cache/<demo_id>/` and its summary KPIs are
        recomputed from the cached per-frame data into `meta.json`.

    --regenerate STEM
        Re-runs the full KeypointPipeline on the matching video under
        `data/matches/` and writes outputs straight into
        `demo_cache/<demo_id>/`. Used when `output/processed_*` doesn't
        yet exist for a bundled video. Requires the model weights from
        `app/streamlit_app.py:MODEL_PATHS` to be present.

Other flags:

    --source STEM        Restrict re-bundling to specific stems (repeatable).
    --dry-run            Print the plan, write nothing.
    --list               Just list demos already in the cache, exit.

Exits 0 on success, 1 on user-fixable setup error, 2 on partial (some
demos skipped). Idempotent: re-running updates changed files only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))

OUTPUT_BASE = _PROJECT_ROOT / "output"
DEMO_CACHE_DIR = _PROJECT_ROOT / "demo_cache"
MATCHES_DIR = _PROJECT_ROOT / "data" / "matches"
SUPPORTED_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov", ".mkv")

VIDEO_BASENAMES = (
    "final_draft.mp4",
    "deep_analysis.mp4",
    "full_pitch_debug_map.mp4",
    "annotated_video.mp4",
    "keypoint_annotations.mp4",
)
REQUIRED_DATA_FILES = (
    "data/game_data.npz",
    "data/analytics_data.json",
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _safe_stem(stem: str) -> str:
    """Same sanitisation the Streamlit pipeline uses for output dirs."""
    out = []
    for c in stem:
        if c.isalnum() or c in " _-":
            out.append(c)
        else:
            out.append("_")
    return "".join(out).strip()


def _demo_id_from_stem(stem: str) -> str:
    """Stable id used as the on-disk demo directory name."""
    cleaned = _safe_stem(stem)
    return cleaned.replace(" ", "-") or "demo"


def _iter_processed_runs() -> list[Path]:
    if not OUTPUT_BASE.exists():
        return []
    runs = []
    for sub in sorted(OUTPUT_BASE.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("processed_"):
            continue
        # A run is "complete" when it has the 5 MP4s and the data files.
        if all((sub / name).exists() for name in VIDEO_BASENAMES) and all(
            (sub / name).exists() for name in REQUIRED_DATA_FILES
        ):
            runs.append(sub)
    return runs


def _iter_incomplete_processed_runs() -> list[Path]:
    """Return processed_* dirs that have the MP4s but are missing data files.

    These need `--regenerate` to gain the per-frame analytics. The re-bundle
    loop surfaces them so the user understands why a folder is being
    skipped and what to do next.
    """
    if not OUTPUT_BASE.exists():
        return []
    out: list[Path] = []
    for sub in sorted(OUTPUT_BASE.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("processed_"):
            continue
        if not all((sub / name).exists() for name in VIDEO_BASENAMES):
            continue
        if not all((sub / name).exists() for name in REQUIRED_DATA_FILES):
            out.append(sub)
    return out


def _iter_match_videos() -> list[Path]:
    """List every video file under data/matches/ that we can process."""
    if not MATCHES_DIR.exists():
        return []
    out: list[Path] = []
    for f in sorted(MATCHES_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(f)
    return out


def _human_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ─── Summary KPIs from cached data ────────────────────────────────────────────
def _compute_meta_from_cached(demo_dir: Path) -> dict:
    """Recompute the meta.json summary KPIs by loading the cached data.

    Uses the same `GameAnalyzer` the dashboard uses so the numbers match
    what the picker card displays vs. what each tab will show.
    """
    try:
        import numpy as np
        from game_analyzer import GameAnalyzer
    except Exception as exc:
        return {"_warning": f"could not import GameAnalyzer: {exc}"}

    game_data_path = demo_dir / "data" / "game_data.npz"
    if not game_data_path.exists():
        return {}

    game_data = _load_game_data(game_data_path)
    if not game_data:
        return {}

    meta: dict = {"schema_version": 1}
    try:
        poss = GameAnalyzer.compute_possession(game_data, "Team 1", "Team 2")
        meta["possession"] = {
            "team1_pct": round(float(poss.get("team1_possession_pct", 0.0)), 1),
            "team2_pct": round(float(poss.get("team2_possession_pct", 0.0)), 1),
            "team1_frames": int(poss.get("team1_frames", 0)),
            "team2_frames": int(poss.get("team2_frames", 0)),
        }
    except Exception as exc:
        meta["possession_error"] = str(exc)

    ball_frames = sum(1 for e in game_data if e.get("ball_position") is not None)
    total_frames = len(game_data)
    if total_frames > 0:
        meta["ball_detection_pct"] = round(100.0 * ball_frames / total_frames, 1)
    meta["processed_frames"] = int(total_frames)

    t1 = GameAnalyzer.dominant_team_bgr(game_data, team=0)
    t2 = GameAnalyzer.dominant_team_bgr(game_data, team=1)
    if t1 is not None:
        meta["team1_bgr"] = [int(x) for x in t1]
    if t2 is not None:
        meta["team2_bgr"] = [int(x) for x in t2]

    # Source video fps / total_frames for the picker header.
    final = demo_dir / "videos" / "final_draft.mp4"
    if final.exists():
        meta["source_video"] = _guess_source_video_name(demo_dir)
        try:
            from frame_scrubber import get_video_meta
            vm = get_video_meta(str(final))
            if vm is not None:
                fps, total, _w, _h = vm
                meta["fps"] = float(fps)
                meta["total_frames"] = int(total)
        except Exception:
            pass
    return meta


def _load_game_data(path: Path) -> list[dict]:
    """Mirror of demo_loader._load_game_data kept local to avoid coupling."""
    try:
        from demo_loader import load_demo  # noqa: F401  (smoke test)
    except Exception:
        pass
    # Defer the actual loading to demo_loader for a single source of truth.
    import numpy as np
    with np.load(path, allow_pickle=True) as npz:
        cols = {k: np.asarray(npz[k]) for k in npz.files}
    if "frame_idx" not in cols:
        return []
    n = int(cols["frame_idx"].shape[0])
    out: list[dict] = []

    def _row_2d(key, empty_shape, dtype):
        if key not in cols:
            return np.empty(empty_shape, dtype=dtype)
        arr = np.asarray(cols[key][i])
        if arr.ndim == 2:
            return arr.astype(dtype, copy=False)
        return np.empty(empty_shape, dtype=dtype)

    def _row_nullable_int(key):
        NONE_SENT = np.asarray([-1], dtype=np.int32)
        if key not in cols:
            return None
        arr = np.asarray(cols[key][i])
        if arr.size == NONE_SENT.size and np.array_equal(
            arr.astype(np.int32, copy=False).reshape(-1), NONE_SENT.reshape(-1)
        ):
            return None
        if arr.size == 0:
            return np.empty((0,), dtype=np.int32)
        return arr.astype(np.int32, copy=False)

    for i in range(n):
        bp = None
        if "ball_position" in cols:
            arr = np.asarray(cols["ball_position"][i]).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[0]) and np.isfinite(arr[1]):
                bp = arr[:2].astype(np.float32, copy=False)
        out.append({
            "frame_idx": int(np.asarray(cols["frame_idx"][i]).reshape(-1)[0]),
            "player_positions": _row_2d("player_positions", (0, 2), np.float32),
            "ball_position": bp,
            "track_ids": _row_nullable_int("track_ids"),
            "team_ids": _row_nullable_int("team_ids"),
        })
    return out


def _guess_source_video_name(demo_dir: Path) -> str:
    """Recover the original video filename from the demo_id when possible."""
    # demo_id strips most punctuation; this is only used for display.
    return demo_dir.name


# ─── Re-bundle mode ───────────────────────────────────────────────────────────
def re_bundle(args) -> int:
    runs = _iter_processed_runs()
    if args.source:
        wanted = {_safe_stem(s) for s in args.source}
        runs = [r for r in runs if _safe_stem(r.name[len("processed_"):]) in wanted]
    if not runs:
        # Distinguish "no runs at all" from "no runs with data/". The
        # latter is the common case for runs that pre-date the demo-cache
        # feature and need `--regenerate` to gain per-frame analytics.
        incomplete = _iter_incomplete_processed_runs()
        if args.source:
            wanted = {_safe_stem(s) for s in args.source}
            incomplete = [r for r in incomplete
                          if _safe_stem(r.name[len("processed_"):]) in wanted]
        if incomplete:
            print(f"[setup_demo_cache] {len(incomplete)} processed run(s) are missing data/ — re-process them:")
            for r in incomplete:
                print(f"    - {r.name[len('processed_'):]}")
            print("    python scripts/setup_demo_cache.py --regenerate --source <stem>")
            print("    python scripts/setup_demo_cache.py --all")
            return 1
        print("[setup_demo_cache] No completed runs found in output/.")
        print("    Run the pipeline at least once, or pass --regenerate <stem>")
        print("    to populate the cache from scratch.")
        return 1

    added: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []

    for run_dir in runs:
        stem = run_dir.name[len("processed_"):]
        demo_id = _demo_id_from_stem(stem)
        demo_dir = DEMO_CACHE_DIR / demo_id
        already = demo_dir.exists()
        verb = "update" if already else "add  "
        if args.dry_run:
            print(f"    {verb} {demo_id} (from {run_dir.name})")
            continue
        try:
            _populate_demo_from_run(run_dir, demo_dir, stem)
            (added if not already else updated).append(demo_id)
            print(f"    {verb} {demo_id}")
        except Exception as exc:
            skipped.append(demo_id)
            print(f"    SKIP {demo_id}: {exc}")

    # Surface runs that have MP4s but no data/ folder so the user knows
    # why they're being skipped and how to fix it.
    incomplete = _iter_incomplete_processed_runs()
    if args.source:
        wanted = {_safe_stem(s) for s in args.source}
        incomplete = [r for r in incomplete
                      if _safe_stem(r.name[len("processed_"):]) in wanted]

    if not args.dry_run:
        _rebuild_top_level_manifest()
    print()
    print(f"    added:   {len(added)}")
    print(f"    updated: {len(updated)}")
    print(f"    skipped: {len(skipped)}")
    print(f"    cache:   {DEMO_CACHE_DIR}")
    if incomplete:
        print()
        print(f"    {len(incomplete)} processed run(s) have no data/ folder and were skipped:")
        for r in incomplete:
            print(f"        - {r.name[len('processed_'):]}")
        print("    Re-process them with:  python scripts/setup_demo_cache.py --regenerate --source <stem>")
        print("    Or process every video at once with:  python scripts/setup_demo_cache.py --all")
    return 2 if (skipped or incomplete) else 0


def _populate_demo_from_run(run_dir: Path, demo_dir: Path, stem: str) -> None:
    """Copy videos + data files from a completed processed run into the cache."""
    videos_dir = demo_dir / "videos"
    data_dir = demo_dir / "data"
    videos_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in VIDEO_BASENAMES:
        src = run_dir / name
        dst = videos_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing in source: {src}")
        _copy_file(src, dst)

    # REQUIRED_DATA_FILES are names relative to the processed run (i.e.
    # `data/game_data.npz`); strip the leading `data/` when copying so
    # they end up at `<demo_dir>/data/game_data.npz`, not
    # `<demo_dir>/data/data/game_data.npz`.
    for name in REQUIRED_DATA_FILES:
        src = run_dir / name
        flat = name[len("data/"):] if name.startswith("data/") else name
        dst = data_dir / flat
        if not src.exists():
            raise FileNotFoundError(f"missing in source: {src}")
        _copy_file(src, dst)

    meta = _compute_meta_from_cached(demo_dir)
    meta.setdefault("schema_version", 1)
    meta["source_stem"] = stem
    meta["source_run_dir"] = str(run_dir)
    with open(demo_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    files = [f"videos/{n}" for n in VIDEO_BASENAMES] + [
        "data/game_data.npz",
        "data/analytics_data.json",
        "meta.json",
        "manifest.json",
    ]
    manifest = {
        "schema_version": 1,
        "id": demo_dir.name,
        "title": _title_from_stem(stem),
        "source_video": meta.get("source_video") or stem,
        "fps": float(meta.get("fps") or 30.0),
        "total_frames": int(meta.get("total_frames") or 0),
        "files": files,
    }
    with open(demo_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _title_from_stem(stem: str) -> str:
    cleaned = stem.replace("_", " ").strip()
    return cleaned[:120] or stem


def _copy_file(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return  # up to date
    shutil.copyfile(src, dst)


def _rebuild_top_level_manifest() -> None:
    if not DEMO_CACHE_DIR.exists():
        return
    demos = []
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
        demos.append({
            "id": data.get("id") or sub.name,
            "title": data.get("title") or sub.name,
            "source_video": data.get("source_video") or "",
            "fps": data.get("fps"),
            "total_frames": data.get("total_frames"),
            "files": data.get("files") or [],
        })
    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_CACHE_DIR / "manifest.json", "w") as f:
        json.dump({"schema_version": 1, "demos": demos}, f, indent=2)


# ─── Regenerate mode ──────────────────────────────────────────────────────────
def regenerate(args) -> int:
    if not args.source:
        print("[setup_demo_cache] --regenerate requires --source <stem>")
        return 1

    added: list[str] = []
    failed: list[str] = []

    for stem in args.source:
        rc = _regenerate_one(stem, args)
        if rc == 0:
            added.append(stem)
        else:
            failed.append(stem)

    if added:
        _rebuild_top_level_manifest()

    print()
    print(f"    regenerated: {len(added)}")
    print(f"    failed:      {len(failed)}")
    if failed:
        print("    failed stems:")
        for s in failed:
            print(f"        - {s}")
    return 2 if failed else 0


def _regenerate_one(stem: str, args) -> int:
    """Run the pipeline for a single source video and stamp its demo bundle.

    Returns 0 on success, 1 on failure. Never raises — failures are
    surfaced via the return code so the batch loop can continue.
    """
    src_video = _find_match_video(stem)
    if src_video is None:
        print(f"    SKIP '{stem}': no video matching in {MATCHES_DIR}")
        return 1

    demo_id = _demo_id_from_stem(stem)
    demo_dir = DEMO_CACHE_DIR / demo_id
    (demo_dir / "videos").mkdir(parents=True, exist_ok=True)
    (demo_dir / "data").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"    would re-run pipeline on {src_video.name} → {demo_dir}")
        return 0

    try:
        from keypoint_pipeline import KeypointPipeline
        from config import MODEL_PATHS
    except Exception as exc:
        print(f"    SKIP '{stem}': could not import pipeline: {exc}")
        return 1

    missing = [n for n, p in MODEL_PATHS.items() if not Path(p).exists()]
    if missing:
        print(f"    SKIP '{stem}': missing model weights: {missing}")
        return 1

    pipeline = KeypointPipeline(
        keypoint_model_path=MODEL_PATHS["keypoint"],
        player_model_path=MODEL_PATHS["player"],
        seg_model_path=MODEL_PATHS["seg"],
        ball_model_path=MODEL_PATHS["ball"],
    )
    print(f"    running pipeline on {src_video.name} ...")
    try:
        for _ in pipeline.process_video(
            source_video_path=str(src_video),
            output_dir=str(demo_dir),
            persist_data=True,
        ):
            pass
    except Exception as exc:
        print(f"    FAIL '{stem}': pipeline error: {exc}")
        return 1

    # Re-stamp the demo manifest / meta now that the data is on disk.
    try:
        _populate_demo_from_run(
            run_dir=demo_dir,  # source == destination: pipeline wrote everything here
            demo_dir=demo_dir,
            stem=stem,
        )
    except Exception as exc:
        print(f"    FAIL '{stem}': could not stamp demo: {exc}")
        return 1

    print(f"    OK    '{demo_id}'")
    return 0


def _find_match_video(stem: str) -> Path | None:
    if not MATCHES_DIR.exists():
        return None
    for f in MATCHES_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and f.stem == stem:
            return f
    return None


# ─── --all mode ───────────────────────────────────────────────────────────────
def regenerate_all(args) -> int:
    """Process every video under data/matches/ into demo_cache/.

    Uses the same code path as --regenerate so the pipeline runs end-to-end
    (including writing per-frame data) — useful when `output/processed_*`
    is missing or stale. Each video is independent: a failure on one does
    not abort the rest.
    """
    videos = _iter_match_videos()
    if not videos:
        print(f"[setup_demo_cache] No videos found under {MATCHES_DIR}")
        return 1

    if args.dry_run:
        for v in videos:
            stem = v.stem
            demo_id = _demo_id_from_stem(stem)
            print(f"    would re-run pipeline on {v.name} → demo_cache/{demo_id}/")
        return 0

    stems = [v.stem for v in videos]
    added: list[str] = []
    failed: list[str] = []

    print(f"[setup_demo_cache] --all: processing {len(stems)} video(s) from {MATCHES_DIR}")
    for stem in stems:
        rc = _regenerate_one(stem, args)
        if rc == 0:
            added.append(stem)
        else:
            failed.append(stem)

    _rebuild_top_level_manifest()
    print()
    print(f"    regenerated: {len(added)}")
    print(f"    failed:      {len(failed)}")
    if added:
        print("    added demos:")
        for s in added:
            print(f"        - {_demo_id_from_stem(s)}")
    if failed:
        print("    failed stems:")
        for s in failed:
            print(f"        - {s}")
    return 2 if failed else 0


# ─── List mode ────────────────────────────────────────────────────────────────
def list_existing() -> int:
    if not DEMO_CACHE_DIR.exists():
        print(f"[setup_demo_cache] {DEMO_CACHE_DIR} does not exist yet")
        return 0
    rows: list[tuple[str, str, str, int, str]] = []
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
        files = data.get("files") or []
        missing = [f for f in files if not (sub / f).exists()]
        status = "OK" if not missing else f"MISSING {len(missing)}"
        rows.append((
            data.get("id") or sub.name,
            data.get("title") or "",
            status,
            _dir_size(sub),
            str(sub),
        ))
    if not rows:
        print("[setup_demo_cache] cache is empty (no manifest.json files yet)")
        return 0
    name_w = max(len(r[0]) for r in rows)
    title_w = min(50, max(len(r[1]) for r in rows))
    print(f"{'id'.ljust(name_w)}  {'title'.ljust(title_w)}  {'status'.ljust(12)}  size")
    print("-" * (name_w + title_w + 32))
    for r in rows:
        title = r[1][:title_w]
        print(
            f"{r[0].ljust(name_w)}  {title.ljust(title_w)}  "
            f"{r[2].ljust(12)}  {_human_mb(r[3])}"
        )
    return 0


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Populate PitchSense's demo cache with pre-cached output bundles.",
    )
    parser.add_argument("--source", action="append", default=[],
                        help="Restrict to a specific source video stem (repeatable).")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-run the full pipeline on the matching source video(s) "
                             "instead of re-bundling from an existing run.")
    parser.add_argument("--all", dest="regenerate_all", action="store_true",
                        help="Re-run the pipeline on every video under data/matches/ "
                             "and write each into demo_cache/<demo_id>/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan, write nothing.")
    parser.add_argument("--list", action="store_true",
                        help="List demos already in the cache and exit.")
    args = parser.parse_args(argv)

    if args.list and not (args.source or args.regenerate or args.regenerate_all):
        return list_existing()
    if args.list:
        # Allow `--list --source X` for a per-demo summary, but treat it as
        # a no-op for now.
        return list_existing()

    if args.regenerate_all:
        if args.source:
            print("[setup_demo_cache] --all ignores --source (it processes every match)")
        return regenerate_all(args)
    if args.regenerate:
        return regenerate(args)
    return re_bundle(args)


if __name__ == "__main__":
    raise SystemExit(main())