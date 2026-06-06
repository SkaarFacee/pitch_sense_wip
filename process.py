"""
Unified SoccerNet Keypoints & Pitch Processing Pipeline (No Download).

Process an already-downloaded SoccerNet calibration dataset to produce:
  - Pitch object detections (green-area segmentation)
  - Field keypoints from line-intersection calculations
  - Unified JSON annotations
  - Ultralytics YOLO pose-format labels
  - Visualisation images
  - dataset.yaml configuration

The ``--task`` argument controls which SoccerNet task subdirectory is used
(e.g. ``"calibration"`` or ``"calibration-2023"``).

Usage:
    python process_existing_dataset.py --dataset_path /path/to/SoccerNet/Data
    python process_existing_dataset.py --dataset_path /path/to/SoccerNet/Data --task calibration-2023
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import tqdm

# ── Project-level imports ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from Data_utils.SoccerNet_Keypoints.get_pitch_object import PitchDetector
from Data_utils.SoccerNet_Keypoints.line_intersections import LineIntersectionCalculator


# ======================================================================
# 1.  Dataset discovery & organisation
# ======================================================================

def discover_soccernet_structure(dataset_path: Path, task: str = "calibration") -> Dict[str, Path]:
    """Auto-detect the directory layout under ``dataset_path / <task>``.

    The ``task`` parameter controls which SoccerNet calibration subdirectory
    is used (e.g. ``"calibration"``, ``"calibration-2023"``).

    Supports three common SoccerNet layouts:

    *Layout A* – raw download, organised by match folders::

        {task}/
            train/
                <match>/
                    images/
                    calibration.json
            test/
            valid/

    *Layout B* – already pre-processed (flat JSON + images)::

        {task}/
            soccernet_calibration_annotations/
                train/*.json
                test/*.json
                valid/*.json
            images/
                train/*.jpg
                test/*.jpg
                valid/*.jpg

    *Layout C* – flat files directly inside split folders (SoccerNet v2 bundles)::

        {task}/
            train/
                *.jpg
                *.json
            test/
                *.jpg
                *.json
            valid/
                *.jpg
                *.json

    Returns
    -------
    dict with keys ``"images_root"``, ``"annot_root"``, or raises ``FileNotFoundError``.
    """
    calib = dataset_path / task
    if not calib.exists():
        raise FileNotFoundError(
            f"Expected a '{task}' subdirectory under {dataset_path}, "
            f"but it does not exist.  Please point --dataset_path at the "
            f"directory that CONTAINS the '{task}' folder."
        )

    # ── Layout B (already flat) ────────────────────────────────────
    flat_images = calib / "images"
    flat_annot  = calib / "soccernet_calibration_annotations"
    if flat_images.exists() and flat_annot.exists():
        print(f"  ✓ Detected Layout B – images & annotations already organised under '{task}'.")
        return {"images_root": flat_images, "annot_root": flat_annot}

    # ── Check that at least one split folder exists ─────────────────
    splits_found = [s for s in ("train", "test", "valid") if (calib / s).exists()]
    if not splits_found:
        raise FileNotFoundError(
            f"No recognised data layout under {calib}.  "
            f"Expected either:\n"
            f"  • Layout A: {calib}/{{train,test,valid}}/<match>/\n"
            f"  • Layout B: {calib}/images/ + {calib}/soccernet_calibration_annotations/"
            f"  • Layout C: {calib}/{{train,test,valid}}/*.jpg + *.json"
        )

    # ── Layout C (flat files in split dirs) vs Layout A (match subdirs) ──
    # Check first split: if its children are mostly files (not dirs), it's Layout C.
    first_split = splits_found[0]
    first_split_items = list((calib / first_split).iterdir())
    first_split_dirs = [p for p in first_split_items if p.is_dir()]
    first_split_files = [p for p in first_split_items if p.is_file()]

    # If there are few or no subdirectories and plenty of image/json files, it's Layout C.
    if len(first_split_dirs) <= 1 and any(
        p.suffix.lower() in (".jpg", ".jpeg", ".png", ".json") for p in first_split_files
    ):
        print(f"  ✓ Detected Layout C – flat files in split folders under '{task}'. Organising ...")
        return _flatten_flat_split_structure(calib, splits_found)

    # ── Layout A (raw SoccerNet with match subdirs) ──────────────────
    print(f"  ✓ Detected Layout A – raw SoccerNet structure under '{task}'. Flattening ...")
    return _flatten_match_subdir_structure(calib, splits_found)


def _flatten_match_subdir_structure(calib: Path, splits: List[str]) -> Dict[str, Path]:
    """Flatten Layout A (match subdirectories) into Layout B.

    Before::

        {task}/train/<match_id>/images/*.jpg  +  <match_id>/*.json

    After::

        {task}/images/train/*.jpg
        {task}/soccernet_calibration_annotations/train/*.json
    """
    images_dir = calib / "images"
    annot_dir  = calib / "soccernet_calibration_annotations"

    for split in splits:
        split_path = calib / split
        if not split_path.is_dir():
            continue

        # Find all match subdirectories
        match_dirs = [d for d in split_path.iterdir() if d.is_dir()]

        # Collect images & JSON files
        for match_dir in match_dirs:
            # ── Images ──────────────────────────────────────────
            match_images = match_dir / "images"
            if match_images.exists():
                out_img = images_dir / split
                out_img.mkdir(parents=True, exist_ok=True)
                for img_file in match_images.glob("*.[jJ][pP][gG]"):
                    dest = out_img / img_file.name
                    if not dest.exists():
                        shutil.copy2(img_file, dest)

            # ── Annotation JSON ─────────────────────────────────
            for json_file in match_dir.glob("*.json"):
                out_json = annot_dir / split
                out_json.mkdir(parents=True, exist_ok=True)
                dest = out_json / json_file.name
                if not dest.exists():
                    shutil.copy2(json_file, dest)

    print(f"  -> Images flattened to:   {images_dir}")
    print(f"  -> Annotations flattened: {annot_dir}")
    return {"images_root": images_dir, "annot_root": annot_dir}


def _flatten_flat_split_structure(calib: Path, splits: List[str]) -> Dict[str, Path]:
    """Organise Layout C (flat files in split dirs) into Layout B.

    Before::

        {task}/train/*.jpg + *.json
        {task}/test/*.jpg  + *.json

    After::

        {task}/images/train/*.jpg
        {task}/soccernet_calibration_annotations/train/*.json
    """
    images_dir = calib / "images"
    annot_dir  = calib / "soccernet_calibration_annotations"

    for split in splits:
        split_path = calib / split
        if not split_path.is_dir():
            continue

        # ── Images ──────────────────────────────────────────────
        out_img = images_dir / split
        out_img.mkdir(parents=True, exist_ok=True)
        for img_file in split_path.glob("*.[jJ][pP][gG]"):
            dest = out_img / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)

        # ── Annotation JSON ─────────────────────────────────────
        out_json = annot_dir / split
        out_json.mkdir(parents=True, exist_ok=True)
        for json_file in split_path.glob("*.json"):
            dest = out_json / json_file.name
            if not dest.exists():
                shutil.copy2(json_file, dest)

    print(f"  -> Images copied to:   {images_dir}")
    print(f"  -> Annotations copied: {annot_dir}")
    return {"images_root": images_dir, "annot_root": annot_dir}


# ======================================================================
# 2.  Annotation helpers  (ported from process_images.py)
# ======================================================================

def create_ultralytics_annotation(
    pitch_data: Dict,
    keypoints: Dict,
    image_shape: Tuple[int, int],
) -> str:
    """Ultralytics YOLO pose-format string.

    Format: ``<class> <cx> <cy> <w> <h> <px1> <py1> <pv1> ... <pxN> <pyN> <pvN>``

    Visibility flag: ``2`` = visible, ``0`` = not visible.
    """
    parts = [
        "0",
        f"{pitch_data['center_x']:.6f}",
        f"{pitch_data['center_y']:.6f}",
        f"{pitch_data['width']:.6f}",
        f"{pitch_data['height']:.6f}",
    ]

    kp_order = [
        "0_sideline_top_left",
        "1_big_rect_left_top_pt1",
        "2_big_rect_left_top_pt2",
        "3_big_rect_left_bottom_pt1",
        "4_big_rect_left_bottom_pt2",
        "5_small_rect_left_top_pt1",
        "6_small_rect_left_top_pt2",
        "7_small_rect_left_bottom_pt1",
        "8_small_rect_left_bottom_pt2",
        "9_sideline_bottom_left",
        "10_left_semicircle_right",
        "11_center_line_top",
        "12_center_line_bottom",
        "13_center_circle_top",
        "14_center_circle_bottom",
        "15_field_center",
        "16_sideline_top_right",
        "17_big_rect_right_top_pt1",
        "18_big_rect_right_top_pt2",
        "19_big_rect_right_bottom_pt1",
        "20_big_rect_right_bottom_pt2",
        "21_small_rect_right_top_pt1",
        "22_small_rect_right_top_pt2",
        "23_small_rect_right_bottom_pt1",
        "24_small_rect_right_bottom_pt2",
        "25_sideline_bottom_right",
        "26_right_semicircle_left",
        "27_center_circle_left",
        "28_center_circle_right",
    ]

    for name in kp_order:
        if name in keypoints:
            x, y = keypoints[name]
            parts.extend([f"{x:.6f}", f"{y:.6f}", "2"])
        else:
            parts.extend(["0.0", "0.0", "0"])

    return " ".join(parts)


def create_unified_visualization(
    image_path: str,
    pitch_data: Dict,
    keypoints: Dict,
    lines: Dict,
    output_path: str,
) -> None:
    """Overlay pitch bounding-box, original lines, and calculated keypoints."""
    image = cv2.imread(image_path)
    if image is None:
        return

    h, w = image.shape[:2]

    # Pitch bounding box
    x_min = int(pitch_data["x_min"] * w)
    y_min = int(pitch_data["y_min"] * h)
    x_max = int(pitch_data["x_max"] * w)
    y_max = int(pitch_data["y_max"] * h)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.putText(image, "Pitch", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Original lines (green)
    if lines:
        for name, pts in lines.items():
            if len(pts) >= 2 and name not in ("Circle central", "Circle left", "Circle right"):
                p1 = (int(pts[0]["x"] * w), int(pts[0]["y"] * h))
                p2 = (int(pts[1]["x"] * w), int(pts[1]["y"] * h))
                cv2.line(image, p1, p2, (0, 150, 0), 1)

        for circ_name in ("Circle central", "Circle left", "Circle right"):
            if circ_name in lines:
                for pt in lines[circ_name]:
                    cv2.circle(image, (int(pt["x"] * w), int(pt["y"] * h)), 2, (0, 150, 0), -1)

    # Calculated keypoints (red)
    for i, (kp_name, (kx, ky)) in enumerate(keypoints.items()):
        pt = (int(kx * w), int(ky * h))
        cv2.circle(image, pt, 6, (0, 0, 255), -1)
        num = kp_name.split("_")[0]
        cv2.putText(image, num, (pt[0] + 8, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(output_path, image)


# ======================================================================
# 3.  Main processing pipeline
# ======================================================================

def process_unified_soccernet_dataset(
    images_root: Path,
    annot_root: Path,
    output_base: Optional[Path] = None,
    splits: Optional[List[str]] = None,
    task: str = "calibration",
) -> None:
    """Run the full pipeline on an already-downloaded SoccerNet dataset.

    Parameters
    ----------
    images_root:
        Directory containing ``{split}/*.jpg`` sub-folders.
    annot_root:
        Directory containing ``{split}/*.json`` sub-folders.
    output_base:
        Where to write results (default: ``images_root.parent / 'unified_output_{task}'``).
    splits:
        Which dataset splits to process (default: ``["train", "test", "valid"]``).
    task:
        SoccerNet task name (e.g. ``"calibration"``, ``"calibration-2023"``).
    """
    if splits is None:
        splits = ["train", "test", "valid"]

    # ── Output directories ──────────────────────────────────────────
    if output_base is None:
        output_base = images_root.parent / f"unified_output_{task}"

    json_dir      = output_base / "annotations_json"
    images_dir    = output_base / "processed_images"
    yolo_dir      = output_base / "yolo_labels"

    for d in (output_base, json_dir, images_dir, yolo_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Initialise processors ───────────────────────────────────────
    calculator    = LineIntersectionCalculator()
    pitch_detector = PitchDetector()

    # ── Iterate over splits ─────────────────────────────────────────
    for split in splits:
        img_split  = images_root / split
        ann_split  = annot_root / split

        if not img_split.exists():
            print(f"  Warning: Skipping '{split}' - images folder not found: {img_split}")
            continue
        if not ann_split.exists():
            print(f"  Warning: Skipping '{split}' - annotations folder not found: {ann_split}")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing split: {split}")
        print(f"  Images:      {img_split}")
        print(f"  Annotations: {ann_split}")

        # Create per-split output sub-folders
        for d in (json_dir, images_dir, yolo_dir):
            (d / split).mkdir(parents=True, exist_ok=True)

        json_files = sorted(ann_split.glob("*.json"))
        print(f"  Found {len(json_files)} annotation file(s).")

        for json_path in tqdm.tqdm(json_files, desc=f"  {split}"):
            base_name  = json_path.stem

            # Try common image extensions
            image_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = img_split / f"{base_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                # Some SoccerNet JSON files use a slightly different naming;
                # try matching by checking all images against the stem.
                for img_file in img_split.iterdir():
                    if img_file.stem in base_name or base_name in img_file.stem:
                        image_path = img_file
                        break

            if image_path is None:
                tqdm.tqdm.write(f"  Warning: No image found for {json_path.name}, skipping.")
                continue

            try:
                # 1. Calculate keypoints from line annotations
                calculator.load_soccernet_data(str(json_path))
                keypoints, lines = calculator.calculate_field_keypoints()

                # 2. Detect pitch object
                pitch_result = pitch_detector.detect_pitch_from_image(str(image_path))
                if pitch_result is None:
                    tqdm.tqdm.write(f"  Warning: Pitch detection failed: {image_path.name}")
                    continue

                pitch_data  = pitch_result["pitch_detection"]
                img_shape   = (
                    pitch_result["image_shape"]["height"],
                    pitch_result["image_shape"]["width"],
                )

                # 3. Unified JSON annotation
                unified = {
                    "image_info": {
                        "file_name": image_path.name,
                        "path": str(image_path),
                        "width": img_shape[1],
                        "height": img_shape[0],
                    },
                    "pitch_object": pitch_data,
                    "keypoints": keypoints,
                    "original_lines": lines,
                    "dataset_split": split,
                    "total_keypoints": len(keypoints),
                    "annotation_format": "SoccerNet_unified_v1",
                }
                with open(json_dir / split / f"{base_name}.json", "w", encoding="utf-8") as f:
                    json.dump(unified, f, indent=2)

                # 4. Ultralytics YOLO label
                yolo_label = create_ultralytics_annotation(pitch_data, keypoints, img_shape)
                with open(yolo_dir / split / f"{base_name}.txt", "w", encoding="utf-8") as f:
                    f.write(yolo_label + "\n")

                # 5. Visualisation
                vis_path = images_dir / split / f"{base_name}_annotated.jpg"
                create_unified_visualization(
                    str(image_path), pitch_data, keypoints, lines, str(vis_path),
                )

            except Exception as exc:
                tqdm.tqdm.write(f"  Error processing {json_path.name}: {exc}")
                continue

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Processing complete!")
    print(f"  JSON annotations: {json_dir}")
    print(f"  Visualisations:   {images_dir}")
    print(f"  YOLO labels:      {yolo_dir}")

    # ── dataset.yaml ─────────────────────────────────────────────────
    create_yolo_dataset_config(output_base)


def create_yolo_dataset_config(output_base: Path) -> None:
    """Write ``dataset.yaml`` for Ultralytics YOLO training."""
    yaml_content = f"""# SoccerNet Keypoints Dataset Configuration for Ultralytics YOLO
# (auto-generated by process_existing_dataset.py)

path: {output_base.absolute()}
train: yolo_labels/train
val: yolo_labels/valid
test: yolo_labels/test

kpt_shape: [29, 3]   # 29 keypoints, each (x, y, visibility)

names:
  0: pitch

kpt_connections:
  - [0, 1]
  - [1, 2]
  - [2, 3]
  - [3, 4]
  - [9, 0]
  - [16, 26]
  - [11, 12]
  - [13, 14]

keypoint_names:
  0: sideline_top_left
  1: big_rect_left_top_pt1
  2: big_rect_left_top_pt2
  3: big_rect_left_bottom_pt1
  4: big_rect_left_bottom_pt2
  5: small_rect_left_top_pt1
  6: small_rect_left_top_pt2
  7: small_rect_left_bottom_pt1
  8: small_rect_left_bottom_pt2
  9: sideline_bottom_left
  10: left_semicircle_right
  11: center_line_top
  12: center_line_bottom
  13: center_circle_top
  14: center_circle_bottom
  15: field_center
  16: sideline_top_right
  17: big_rect_right_top_pt1
  18: big_rect_right_top_pt2
  19: big_rect_right_bottom_pt1
  20: big_rect_right_bottom_pt2
  21: small_rect_right_top_pt1
  22: small_rect_right_top_pt2
  23: small_rect_right_bottom_pt1
  24: small_rect_right_bottom_pt2
  25: sideline_bottom_right
  26: right_semicircle_left
  27: center_circle_left
  28: center_circle_right

download: false
nc: 1
"""
    yaml_path = output_base / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  Dataset config:     {yaml_path}")

    # README
    readme = """# SoccerNet Keypoints Dataset

## Directory Structure
- `annotations_json/` - Complete JSON annotations with pitch objects & keypoints
- `processed_images/` - Visualisation images
- `yolo_labels/`      - Ultralytics YOLO-format labels
- `dataset.yaml`      - YOLO configuration

## Training with Ultralytics
```python
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, batch=16)
```
"""
    readme_path = output_base / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    print(f"  README:             {readme_path}")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process an already-downloaded SoccerNet calibration dataset "
            "without re-downloading anything."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help=(
            "Path to the SoccerNet Data directory that contains the task "
            "sub-folder (e.g. /path/to/SoccerNet/Data)."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="calibration",
        help=(
            "SoccerNet calibration task name.  This controls which "
            "subdirectory is used under --dataset_path. "
            "Examples: 'calibration' (default), 'calibration-2023'."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Override output directory (default: {task}/unified_output_{task}).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test", "valid"],
        help="Dataset splits to process (default: train test valid).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    SEP = "=" * 60
    print(SEP)
    print("  SoccerNet Keypoints - Processing Pipeline (no download)")
    print(SEP)
    print(f"  Dataset root: {dataset_path}")
    print(f"  Task:         {args.task}")
    print(f"  Splits:       {args.splits}")

    # ── Discover / organise dataset ─────────────────────────────────
    try:
        layout = discover_soccernet_structure(dataset_path, task=args.task)
    except FileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)

    images_root = layout["images_root"]
    annot_root  = layout["annot_root"]

    output_base = Path(args.output_path) if args.output_path else None

    # ── Run pipeline ────────────────────────────────────────────────
    process_unified_soccernet_dataset(
        images_root=images_root,
        annot_root=annot_root,
        output_base=output_base,
        splits=args.splits,
        task=args.task,
    )

    default_output = images_root.parent / f"unified_output_{args.task}"
    print(f"\n  All done!  Results written to: {output_base or default_output}\n")


if __name__ == "__main__":
    main()