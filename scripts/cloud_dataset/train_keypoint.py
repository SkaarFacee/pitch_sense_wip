#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import pathlib
import random
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import torch
from clearml import OutputModel, Task
from ultralytics import YOLO

try:
    from constants import SEED, DEVICE
except Exception:
    SEED = 42
    DEVICE = 0


# =============================================================================
# CONFIG
# =============================================================================

TASK_PROJECT = "PitchSense_v2"
TASK_NAME = "soccernet_calibration_29kp_yolo26n"

CALIBRATION_ROOT = pathlib.Path(
    "/home/aanil/Data/aanil/side/yolo/datasets/Soccernet/calibration-2023"
)

OUTPUT_ROOT = pathlib.Path(
    "/home/aanil/Data/aanil/side/yolo/outputs/29kp_yolo26n"
)

YOLO_DATASET_ROOT = OUTPUT_ROOT / "yolo_dataset"
SAVE_DIR = OUTPUT_ROOT / "saved_models"

MODEL_NAME = "yolo26n-pose.pt"
RUN_NAME = "26s_keypoint"

NUM_KEYPOINTS = 29
EXPECTED_LABEL_LEN = 5 + NUM_KEYPOINTS * 3

TRAIN_EPOCHS = int(os.environ.get("SOCCERNET_EPOCHS", 200))
TRAIN_IMGSZ = int(os.environ.get("SOCCERNET_IMGSZ", 960))
TRAIN_BATCH = int(os.environ.get("SOCCERNET_BATCH", 8))
TRAIN_DEVICE = os.environ.get("SOCCERNET_DEVICE", str(DEVICE))

USE_SYMLINKS = True
REBUILD_YOLO_DATASET = False
MIN_VISIBLE_KEYPOINTS = 4

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}

KEYPOINT_NAMES = {
    0: "sideline_top_left",
    1: "big_rect_left_top_pt1",
    2: "big_rect_left_top_pt2",
    3: "big_rect_left_bottom_pt1",
    4: "big_rect_left_bottom_pt2",
    5: "small_rect_left_top_pt1",
    6: "small_rect_left_top_pt2",
    7: "small_rect_left_bottom_pt1",
    8: "small_rect_left_bottom_pt2",
    9: "sideline_bottom_left",
    10: "left_semicircle_right",
    11: "center_line_top",
    12: "center_line_bottom",
    13: "center_circle_top",
    14: "center_circle_bottom",
    15: "field_center",
    16: "sideline_top_right",
    17: "big_rect_right_top_pt1",
    18: "big_rect_right_top_pt2",
    19: "big_rect_right_bottom_pt1",
    20: "big_rect_right_bottom_pt2",
    21: "small_rect_right_top_pt1",
    22: "small_rect_right_top_pt2",
    23: "small_rect_right_bottom_pt1",
    24: "small_rect_right_bottom_pt2",
    25: "sideline_bottom_right",
    26: "right_semicircle_left",
    27: "center_circle_left",
    28: "center_circle_right",
}


# =============================================================================
# GENERAL UTILS
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_remove(path: pathlib.Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def safe_link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_remove(dst)

    if USE_SYMLINKS:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def get_image_root(split_dir: pathlib.Path) -> pathlib.Path:
    candidates = [
        split_dir / "images",
        split_dir / "image",
        split_dir / "imgs",
        split_dir,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            image_count = sum(
                1
                for p in candidate.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if image_count > 0:
                return candidate

    raise FileNotFoundError(f"No images found under split directory: {split_dir}")


def find_images(split_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    image_root = get_image_root(split_dir)

    out = []
    for p in sorted(image_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(image_root)
            out.append((p, rel))

    return out


def build_json_index(split_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    candidates = [
        split_dir / "jsons",
        split_dir / "json",
        split_dir / "annotations",
        split_dir / "ann",
        split_dir,
    ]

    index: dict[str, pathlib.Path] = {}

    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue

        for p in sorted(root.rglob("*.json")):
            index.setdefault(p.stem, p)

            # Handles frame_001.jpg.json -> frame_001
            nested_stem = pathlib.Path(p.stem).stem
            index.setdefault(nested_stem, p)

    return index


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def read_image(path: pathlib.Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# =============================================================================
# GEOMETRY
# =============================================================================

@dataclass
class LineABC:
    a: float
    b: float
    c: float


def fit_line(points: list[np.ndarray]) -> Optional[LineABC]:
    if len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float64)

    if pts.shape[0] == 2:
        p1, p2 = pts[0], pts[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm = math.hypot(dx, dy)

        if norm < 1e-9:
            return None

        a = dy
        b = -dx
        c = dx * p1[1] - dy * p1[0]
    else:
        center = pts.mean(axis=0)
        centered = pts - center
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]

        a = -direction[1]
        b = direction[0]
        c = -(a * center[0] + b * center[1])

    norm = math.hypot(a, b)
    if norm < 1e-9:
        return None

    return LineABC(a / norm, b / norm, c / norm)


def line_intersection(l1: LineABC, l2: LineABC) -> Optional[np.ndarray]:
    det = l1.a * l2.b - l2.a * l1.b

    if abs(det) < 1e-9:
        return None

    x = (l1.b * l2.c - l2.b * l1.c) / det
    y = (l1.c * l2.a - l2.c * l1.a) / det

    return np.array([x, y], dtype=np.float64)


def farthest_pair(points: list[np.ndarray]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return None

    best = None
    best_dist = -1.0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d > best_dist:
                best_dist = d
                best = (points[i], points[j])

    return best


def get_number(d: dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                return None
    return None


def parse_points(obj: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    if obj is None:
        return points

    if isinstance(obj, dict):
        x = get_number(obj, ["x", "X", "cx", "center_x"])
        y = get_number(obj, ["y", "Y", "cy", "center_y"])

        if x is not None and y is not None:
            return [(x, y)]

        x1 = get_number(obj, ["x1", "X1", "x_min"])
        y1 = get_number(obj, ["y1", "Y1", "y_min"])
        x2 = get_number(obj, ["x2", "X2", "x_max"])
        y2 = get_number(obj, ["y2", "Y2", "y_max"])

        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            return [(x1, y1), (x2, y2)]

        for key in [
            "points",
            "point",
            "vertices",
            "polyline",
            "line",
            "segment",
            "endpoints",
            "coords",
            "coordinates",
            "p1",
            "p2",
            "point1",
            "point2",
            "start",
            "end",
        ]:
            if key in obj:
                points.extend(parse_points(obj[key]))

        if points:
            return points

        for value in obj.values():
            if isinstance(value, (dict, list, tuple)):
                points.extend(parse_points(value))

        return points

    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
            return [(float(obj[0]), float(obj[1]))]

        if len(obj) >= 4 and all(isinstance(v, (int, float)) for v in obj[:4]):
            flat = [float(v) for v in obj]
            for i in range(0, len(flat) - 1, 2):
                points.append((flat[i], flat[i + 1]))
            return points

        for item in obj:
            points.extend(parse_points(item))

        return points

    return points


def canonical_line_name(name: str) -> Optional[str]:
    s = name.lower()
    s = s.replace(".", " ")
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = " ".join(s.split())

    if "side" in s and "top" in s:
        return "side_top"
    if "side" in s and "bottom" in s:
        return "side_bottom"
    if "side" in s and "left" in s:
        return "side_left"
    if "side" in s and "right" in s:
        return "side_right"

    if "touch" in s and "top" in s:
        return "side_top"
    if "touch" in s and "bottom" in s:
        return "side_bottom"
    if "goal" in s and "left" in s and "line" in s:
        return "side_left"
    if "goal" in s and "right" in s and "line" in s:
        return "side_right"

    if "middle" in s or "center line" in s or "centre line" in s:
        return "middle"

    if "big" in s and "rect" in s and "left" in s and "top" in s:
        return "big_left_top"
    if "big" in s and "rect" in s and "left" in s and "bottom" in s:
        return "big_left_bottom"
    if "big" in s and "rect" in s and "left" in s and "main" in s:
        return "big_left_main"

    if "big" in s and "rect" in s and "right" in s and "top" in s:
        return "big_right_top"
    if "big" in s and "rect" in s and "right" in s and "bottom" in s:
        return "big_right_bottom"
    if "big" in s and "rect" in s and "right" in s and "main" in s:
        return "big_right_main"

    if "small" in s and "rect" in s and "left" in s and "top" in s:
        return "small_left_top"
    if "small" in s and "rect" in s and "left" in s and "bottom" in s:
        return "small_left_bottom"
    if "small" in s and "rect" in s and "left" in s and "main" in s:
        return "small_left_main"

    if "small" in s and "rect" in s and "right" in s and "top" in s:
        return "small_right_top"
    if "small" in s and "rect" in s and "right" in s and "bottom" in s:
        return "small_right_bottom"
    if "small" in s and "rect" in s and "right" in s and "main" in s:
        return "small_right_main"

    if "circle" in s and ("central" in s or "center" in s or "centre" in s):
        return "circle_central"
    if "circle" in s and "left" in s:
        return "circle_left"
    if "circle" in s and "right" in s:
        return "circle_right"

    return None


def add_lines_from_container(
    container: Any,
    out: dict[str, list[tuple[float, float]]],
) -> None:
    if isinstance(container, dict):
        for label, value in container.items():
            canon = canonical_line_name(str(label))
            if canon is not None:
                pts = parse_points(value)
                if pts:
                    out.setdefault(canon, []).extend(pts)
        return

    if isinstance(container, list):
        for item in container:
            if not isinstance(item, dict):
                continue

            label = None
            for k in ["label", "name", "class", "category", "line_name", "type"]:
                if k in item:
                    label = str(item[k])
                    break

            if label is None:
                continue

            canon = canonical_line_name(label)
            if canon is None:
                continue

            pts = parse_points(item)
            if pts:
                out.setdefault(canon, []).extend(pts)


def extract_raw_lines(data: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}

    for key in [
        "original_lines",
        "Lines",
        "lines",
        "line_annotations",
        "annotations",
        "field_lines",
        "field_markings",
    ]:
        if key in data:
            add_lines_from_container(data[key], out)

    add_lines_from_container(data, out)

    return out


def to_pixel_point(p: tuple[float, float], width: int, height: int) -> np.ndarray:
    x, y = float(p[0]), float(p[1])

    if -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0:
        return np.array([x * width, y * height], dtype=np.float64)

    return np.array([x, y], dtype=np.float64)


def build_pixel_lines(
    raw_lines: dict[str, list[tuple[float, float]]],
    width: int,
    height: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, LineABC]]:
    pixel_points: dict[str, list[np.ndarray]] = {}
    fitted: dict[str, LineABC] = {}

    for name, pts in raw_lines.items():
        pix = [to_pixel_point(p, width, height) for p in pts]

        if len(pix) < 2:
            continue

        pixel_points[name] = pix
        line = fit_line(pix)

        if line is not None:
            fitted[name] = line

    return pixel_points, fitted


def point_in_image(p: np.ndarray, width: int, height: int, margin: float = 2.0) -> bool:
    x, y = float(p[0]), float(p[1])
    return -margin <= x <= width + margin and -margin <= y <= height + margin


def normalize_pixel_point(
    p: Optional[np.ndarray],
    width: int,
    height: int,
) -> tuple[float, float, int]:
    if p is None:
        return 0.0, 0.0, 0

    if not point_in_image(p, width, height, margin=2.0):
        return 0.0, 0.0, 0

    x = min(max(float(p[0]) / width, 0.0), 1.0)
    y = min(max(float(p[1]) / height, 0.0), 1.0)

    return x, y, 2


def pick_extreme(points: list[np.ndarray], mode: str) -> Optional[np.ndarray]:
    if not points:
        return None

    if mode == "min_x":
        return min(points, key=lambda p: p[0])
    if mode == "max_x":
        return max(points, key=lambda p: p[0])
    if mode == "min_y":
        return min(points, key=lambda p: p[1])
    if mode == "max_y":
        return max(points, key=lambda p: p[1])

    raise ValueError(f"Unknown mode: {mode}")


def calculate_29_keypoints_from_lines(
    raw_lines: dict[str, list[tuple[float, float]]],
    width: int,
    height: int,
) -> list[tuple[float, float, int]]:
    pixel_points, fitted = build_pixel_lines(raw_lines, width, height)

    kp_pix: dict[int, Optional[np.ndarray]] = {
        i: None for i in range(NUM_KEYPOINTS)
    }

    def inter(line_a: str, line_b: str) -> Optional[np.ndarray]:
        if line_a not in fitted or line_b not in fitted:
            return None
        return line_intersection(fitted[line_a], fitted[line_b])

    def add_intersection(idx: int, line_a: str, line_b: str) -> None:
        p = inter(line_a, line_b)
        if p is not None:
            kp_pix[idx] = p

    def add_rect_pair(
        idx1: int,
        idx2: int,
        horizontal_line: str,
        edge_a: str,
        edge_b: str,
    ) -> None:
        pts: list[np.ndarray] = []

        if horizontal_line in fitted:
            for edge in [edge_a, edge_b]:
                if edge in fitted:
                    p = inter(horizontal_line, edge)
                    if p is not None:
                        pts.append(p)

        if len(pts) < 2 and horizontal_line in pixel_points:
            pair = farthest_pair(pixel_points[horizontal_line])
            if pair is not None:
                pts = [pair[0], pair[1]]

        if len(pts) < 2:
            return

        pts = sorted(pts[:2], key=lambda p: float(p[0]))
        kp_pix[idx1] = pts[0]
        kp_pix[idx2] = pts[1]

    add_intersection(0, "side_top", "side_left")
    add_intersection(9, "side_bottom", "side_left")
    add_intersection(16, "side_top", "side_right")
    add_intersection(25, "side_bottom", "side_right")

    add_rect_pair(1, 2, "big_left_top", "side_left", "big_left_main")
    add_rect_pair(3, 4, "big_left_bottom", "side_left", "big_left_main")
    add_rect_pair(17, 18, "big_right_top", "big_right_main", "side_right")
    add_rect_pair(19, 20, "big_right_bottom", "big_right_main", "side_right")

    add_rect_pair(5, 6, "small_left_top", "side_left", "small_left_main")
    add_rect_pair(7, 8, "small_left_bottom", "side_left", "small_left_main")
    add_rect_pair(21, 22, "small_right_top", "small_right_main", "side_right")
    add_rect_pair(23, 24, "small_right_bottom", "small_right_main", "side_right")

    add_intersection(11, "middle", "side_top")
    add_intersection(12, "middle", "side_bottom")

    central_circle_pts = pixel_points.get("circle_central", [])
    if central_circle_pts:
        kp_pix[13] = pick_extreme(central_circle_pts, "min_y")
        kp_pix[14] = pick_extreme(central_circle_pts, "max_y")
        kp_pix[27] = pick_extreme(central_circle_pts, "min_x")
        kp_pix[28] = pick_extreme(central_circle_pts, "max_x")

    center_candidates = []

    if kp_pix[11] is not None and kp_pix[12] is not None:
        center_candidates.append((kp_pix[11] + kp_pix[12]) / 2.0)

    if kp_pix[13] is not None and kp_pix[14] is not None:
        center_candidates.append((kp_pix[13] + kp_pix[14]) / 2.0)

    if kp_pix[27] is not None and kp_pix[28] is not None:
        center_candidates.append((kp_pix[27] + kp_pix[28]) / 2.0)

    if center_candidates:
        kp_pix[15] = np.mean(np.asarray(center_candidates), axis=0)

    left_circle_pts = pixel_points.get("circle_left", [])
    right_circle_pts = pixel_points.get("circle_right", [])

    if left_circle_pts:
        kp_pix[10] = pick_extreme(left_circle_pts, "max_x")

    if right_circle_pts:
        kp_pix[26] = pick_extreme(right_circle_pts, "min_x")

    return [
        normalize_pixel_point(kp_pix[i], width, height)
        for i in range(NUM_KEYPOINTS)
    ]


# =============================================================================
# PROCESSED JSON SUPPORT
# =============================================================================

def parse_processed_keypoints(
    data: dict[str, Any],
) -> Optional[list[tuple[float, float, int]]]:
    kps = data.get("keypoints")
    if not isinstance(kps, dict):
        return None

    out: list[tuple[float, float, int]] = []

    for idx in range(NUM_KEYPOINTS):
        name = KEYPOINT_NAMES[idx]
        possible_keys = [
            f"{idx}_{name}",
            name,
            str(idx),
        ]

        value = None
        for k in possible_keys:
            if k in kps:
                value = kps[k]
                break

        if value is None:
            out.append((0.0, 0.0, 0))
            continue

        coord = parse_points(value)
        if not coord:
            out.append((0.0, 0.0, 0))
            continue

        x, y = coord[0]

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append((float(x), float(y), 2))
        else:
            out.append((0.0, 0.0, 0))

    return out


def parse_processed_pitch_object(
    data: dict[str, Any],
) -> Optional[tuple[float, float, float, float]]:
    obj = data.get("pitch_object")
    if not isinstance(obj, dict):
        return None

    cx = get_number(obj, ["center_x", "cx"])
    cy = get_number(obj, ["center_y", "cy"])
    w = get_number(obj, ["width", "w"])
    h = get_number(obj, ["height", "h"])

    if cx is not None and cy is not None and w is not None and h is not None:
        return clean_bbox(cx, cy, w, h)

    x_min = get_number(obj, ["x_min", "xmin", "x1"])
    y_min = get_number(obj, ["y_min", "ymin", "y1"])
    x_max = get_number(obj, ["x_max", "xmax", "x2"])
    y_max = get_number(obj, ["y_max", "ymax", "y2"])

    if x_min is None or y_min is None or x_max is None or y_max is None:
        return None

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    return clean_bbox(cx, cy, w, h)


# =============================================================================
# PITCH BBOX
# =============================================================================

def clean_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> Optional[tuple[float, float, float, float]]:
    cx = min(max(float(cx), 0.0), 1.0)
    cy = min(max(float(cy), 0.0), 1.0)
    w = min(max(float(w), 0.0), 1.0)
    h = min(max(float(h), 0.0), 1.0)

    if w <= 1e-6 or h <= 1e-6:
        return None

    return cx, cy, w, h


def detect_pitch_bbox_hsv(
    img: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([30, 30, 30], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))

    if area < 0.02 * width * height:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    cx = (x + bw / 2.0) / width
    cy = (y + bh / 2.0) / height
    nw = bw / width
    nh = bh / height

    return clean_bbox(cx, cy, nw, nh)


def bbox_from_keypoints(
    keypoints: list[tuple[float, float, int]],
    pad: float = 0.03,
) -> Optional[tuple[float, float, float, float]]:
    visible = [(x, y) for x, y, v in keypoints if v > 0]

    if len(visible) < 2:
        return None

    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]

    x1 = max(min(xs) - pad, 0.0)
    y1 = max(min(ys) - pad, 0.0)
    x2 = min(max(xs) + pad, 1.0)
    y2 = min(max(ys) + pad, 1.0)

    return clean_bbox(
        cx=(x1 + x2) / 2.0,
        cy=(y1 + y2) / 2.0,
        w=x2 - x1,
        h=y2 - y1,
    )


# =============================================================================
# YOLO LABEL CREATION
# =============================================================================

def format_label_value(v: float | int) -> str:
    if isinstance(v, int):
        return str(v)

    if abs(v - round(v)) < 1e-9 and round(v) in {0, 1, 2}:
        return str(int(round(v)))

    return f"{v:.6f}"


def make_yolo_line(
    bbox: tuple[float, float, float, float],
    keypoints: list[tuple[float, float, int]],
) -> str:
    values: list[float | int] = [0, *bbox]

    for x, y, v in keypoints:
        if v <= 0:
            values.extend([0.0, 0.0, 0])
        else:
            values.extend([x, y, 2])

    if len(values) != EXPECTED_LABEL_LEN:
        raise RuntimeError(
            f"YOLO label has {len(values)} values, expected {EXPECTED_LABEL_LEN}"
        )

    return " ".join(format_label_value(v) for v in values)


def create_label_for_image(
    image_path: pathlib.Path,
    json_path: Optional[pathlib.Path],
) -> tuple[str, dict[str, int]]:
    stats = {
        "json_missing": 0,
        "processed_json_used": 0,
        "line_json_used": 0,
        "empty_labels": 0,
        "low_keypoint_count": 0,
        "bbox_from_json": 0,
        "bbox_from_hsv": 0,
        "bbox_from_keypoints": 0,
        "bbox_full_image": 0,
    }

    img = read_image(image_path)
    height, width = img.shape[:2]

    keypoints: Optional[list[tuple[float, float, int]]] = None
    bbox: Optional[tuple[float, float, float, float]] = None

    if json_path is None or not json_path.exists():
        stats["json_missing"] += 1
        stats["empty_labels"] += 1
        return "", stats

    data = load_json(json_path)

    processed_kps = parse_processed_keypoints(data)

    if processed_kps is not None:
        keypoints = processed_kps
        bbox = parse_processed_pitch_object(data)
        stats["processed_json_used"] += 1

        if bbox is not None:
            stats["bbox_from_json"] += 1
    else:
        raw_lines = extract_raw_lines(data)

        if raw_lines:
            keypoints = calculate_29_keypoints_from_lines(
                raw_lines,
                width,
                height,
            )
            stats["line_json_used"] += 1

    if keypoints is None:
        stats["empty_labels"] += 1
        return "", stats

    visible_count = sum(1 for _, _, v in keypoints if v > 0)

    if visible_count < MIN_VISIBLE_KEYPOINTS:
        stats["low_keypoint_count"] += 1
        stats["empty_labels"] += 1
        return "", stats

    if bbox is None:
        bbox = detect_pitch_bbox_hsv(img)
        if bbox is not None:
            stats["bbox_from_hsv"] += 1

    if bbox is None:
        bbox = bbox_from_keypoints(keypoints)
        if bbox is not None:
            stats["bbox_from_keypoints"] += 1

    if bbox is None:
        bbox = (0.5, 0.5, 1.0, 1.0)
        stats["bbox_full_image"] += 1

    return make_yolo_line(bbox, keypoints), stats


def merge_stats(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in new.items():
        total[k] = total.get(k, 0) + int(v)


def write_dataset_yaml(dataset_root: pathlib.Path) -> pathlib.Path:
    yaml_path = dataset_root / "dataset.yaml"

    yaml_text = f"""path: {dataset_root.resolve()}
train: images/train
val: images/valid
test: images/test

nc: 1
names:
  - pitch

kpt_shape:
  - {NUM_KEYPOINTS}
  - 3

flip_idx:
"""

    for i in range(NUM_KEYPOINTS):
        yaml_text += f"  - {i}\n"

    yaml_text += "\nkeypoint_names:\n"

    for idx, name in KEYPOINT_NAMES.items():
        yaml_text += f"  {idx}: {name}\n"

    yaml_path.write_text(yaml_text)

    return yaml_path


def prepare_local_yolo_dataset() -> tuple[pathlib.Path, dict[str, Any]]:
    if REBUILD_YOLO_DATASET and YOLO_DATASET_ROOT.exists():
        shutil.rmtree(YOLO_DATASET_ROOT)

    YOLO_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    global_stats: dict[str, Any] = {
        "dataset_root": str(YOLO_DATASET_ROOT),
        "splits": {},
    }

    for split_name, split_folder in SPLITS.items():
        split_dir = CALIBRATION_ROOT / split_folder

        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        images = find_images(split_dir)
        json_index = build_json_index(split_dir)

        split_stats: dict[str, int] = {
            "images": 0,
            "non_empty_labels": 0,
            "empty_labels": 0,
        }

        print(f"\nPreparing split: {split_name}")
        print(f"  Split folder: {split_dir}")
        print(f"  Images found: {len(images)}")
        print(f"  JSON files indexed: {len(json_index)}")

        for image_path, rel_image_path in images:
            dst_image = YOLO_DATASET_ROOT / "images" / split_name / rel_image_path
            dst_label = (
                YOLO_DATASET_ROOT
                / "labels"
                / split_name
                / rel_image_path.with_suffix(".txt")
            )

            json_path = json_index.get(image_path.stem)

            label_text, label_stats = create_label_for_image(
                image_path,
                json_path,
            )

            safe_link_or_copy(image_path, dst_image)

            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text(label_text + ("\n" if label_text else ""))

            split_stats["images"] += 1

            if label_text:
                split_stats["non_empty_labels"] += 1
            else:
                split_stats["empty_labels"] += 1

            merge_stats(split_stats, label_stats)

        global_stats["splits"][split_name] = split_stats

        print(f"  Non-empty labels: {split_stats['non_empty_labels']}")
        print(f"  Empty labels: {split_stats['empty_labels']}")
        print(f"  Processed JSON used: {split_stats.get('processed_json_used', 0)}")
        print(f"  Raw line JSON used: {split_stats.get('line_json_used', 0)}")
        print(f"  Missing JSON: {split_stats.get('json_missing', 0)}")
        print(f"  Low-keypoint labels skipped: {split_stats.get('low_keypoint_count', 0)}")

    yaml_path = write_dataset_yaml(YOLO_DATASET_ROOT)
    global_stats["dataset_yaml"] = str(yaml_path)

    return yaml_path, global_stats


# =============================================================================
# TRAINING
# =============================================================================

def copy_best_weights(
    save_dir: pathlib.Path,
    run_name: str,
) -> Optional[pathlib.Path]:
    best_weights = save_dir / run_name / "weights" / "best.pt"
    final_model_path = save_dir / f"{run_name}_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, final_model_path)
        print(f"Saved best model to: {final_model_path}")
        return final_model_path

    print(f"[WARN] best.pt not found at: {best_weights}")
    return None


def main() -> None:
    seed_everything(SEED)

    task = Task.init(
        project_name=TASK_PROJECT,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
    )

    config = {
        "calibration_root": str(CALIBRATION_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "yolo_dataset_root": str(YOLO_DATASET_ROOT),
        "save_dir": str(SAVE_DIR),
        "model_name": MODEL_NAME,
        "run_name": RUN_NAME,
        "num_keypoints": NUM_KEYPOINTS,
        "min_visible_keypoints": MIN_VISIBLE_KEYPOINTS,
        "epochs": TRAIN_EPOCHS,
        "imgsz": TRAIN_IMGSZ,
        "batch": TRAIN_BATCH,
        "device": TRAIN_DEVICE,
        "seed": SEED,
        "use_symlinks": USE_SYMLINKS,
        "rebuild_yolo_dataset": REBUILD_YOLO_DATASET,
        "note": "Dataset stays local. This script does not create or upload a ClearML Dataset.",
    }

    task.connect(config)

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Training device:", TRAIN_DEVICE)
    print("Calibration root:", CALIBRATION_ROOT)
    print("Output root:", OUTPUT_ROOT)
    print("YOLO dataset root:", YOLO_DATASET_ROOT)
    print("Model:", MODEL_NAME)
    print("Run name:", RUN_NAME)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    run_dir = SAVE_DIR / RUN_NAME
    if run_dir.exists():
        print(f"Removing old run directory: {run_dir}")
        shutil.rmtree(run_dir)

    print("\nConverting SoccerNet calibration JSONs to local YOLO pose dataset...")
    yaml_path, conversion_stats = prepare_local_yolo_dataset()

    print("\nDataset conversion stats:")
    print(json.dumps(conversion_stats, indent=2))

    task.upload_artifact("conversion_stats", artifact_object=conversion_stats)
    task.upload_artifact("dataset_yaml_text", artifact_object=yaml_path.read_text())

    print("\nStarting YOLO26n pose training...")
    print("Dataset YAML:", yaml_path)

    model = YOLO(MODEL_NAME)

    train_return = model.train(
        data=str(yaml_path),
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=TRAIN_BATCH,
        device=TRAIN_DEVICE,
        workers=4,
        project=str(SAVE_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,

        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        erasing=0.0,

        hsv_h=0.003,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=1.0,
        translate=0.03,
        scale=0.10,
        perspective=0.0,

        lr0=0.003,
        lrf=0.01,
        cos_lr=True,

        dropout=0.0,
        patience=100,
    )

    run_save_dir = pathlib.Path(
        getattr(train_return, "save_dir", SAVE_DIR / RUN_NAME)
    )
    print("Ultralytics save dir:", run_save_dir)

    final_model_path = copy_best_weights(SAVE_DIR, RUN_NAME)

    if final_model_path is not None and final_model_path.exists():
        output_model = OutputModel(task=task, name=f"{RUN_NAME}_best")
        output_model.update_weights(weights_filename=str(final_model_path))

        task.upload_artifact("best_model_path", artifact_object=str(final_model_path))

        best_model = YOLO(str(final_model_path))

        print("\nRunning validation on valid split...")
        valid_results = best_model.val(
            data=str(yaml_path),
            split="val",
            imgsz=TRAIN_IMGSZ,
            batch=TRAIN_BATCH,
            device=TRAIN_DEVICE,
            project=str(SAVE_DIR),
            name="valid_eval",
            verbose=True,
        )

        print("Valid results:", valid_results)

        print("\nRunning evaluation on test split...")
        test_results = best_model.val(
            data=str(yaml_path),
            split="test",
            imgsz=TRAIN_IMGSZ,
            batch=TRAIN_BATCH,
            device=TRAIN_DEVICE,
            project=str(SAVE_DIR),
            name="test_eval",
            verbose=True,
        )

        print("Test results:", test_results)
    else:
        print("[WARN] No final model path found. Skipping val/test evaluation.")

    print("\nTraining complete.")
    print(f"Local YOLO dataset remains at: {YOLO_DATASET_ROOT}")
    print("No ClearML Dataset was created or uploaded.")

    task.close()


if __name__ == "__main__":
    main()