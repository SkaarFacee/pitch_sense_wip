"""app/config.py — standalone constants for PitchSense.

Imports nothing from the Streamlit dashboard so external tools (e.g.
``scripts/setup_demo_cache.py``) can pull model paths without executing
the entire ``streamlit_app`` module and triggering Streamlit's
"ScriptRunContext" warnings when run outside ``streamlit run``.
"""
from __future__ import annotations

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

# ─── Paths ────────────────────────────────────────────────────────────────────
TEST_DATA_DIR = _PROJECT_ROOT / "data" / "matches"
OUTPUT_BASE = _PROJECT_ROOT / "output"

# ─── Model weights ────────────────────────────────────────────────────────────
MODEL_PATHS: dict[str, str] = {
    "keypoint": str(_PROJECT_ROOT / "models" / "keypoint_model"
                    / "26n_pipeline" / "no_aug" / "weights" / "best.pt"),
    "player":   str(_PROJECT_ROOT / "models" / "player_model" / "best.pt"),
    "seg":      str(_PROJECT_ROOT / "models" / "segmentation" / "best.pt"),
    "ball":     str(_PROJECT_ROOT / "models" / "ball_model" / "yolo26_best.pt"),
}

# ─── Input video discovery ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov", ".mkv")