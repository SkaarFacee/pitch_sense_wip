"""
Train YOLO26 Pose Model from Scratch on Pitch Keypoint Data

Handles variable-size source images safely.

Expected YOLO pose label format:
  class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...

Important:
  - x_center, y_center, width, height must be normalized to [0, 1]
  - keypoint x/y must be normalized to [0, 1]
  - visibility values are usually 0, 1, or 2
  - normalization must be done using each image's own width and height
"""

from ultralytics import YOLO
import torch
from pathlib import Path
from clearml import Task
from PIL import Image
import yaml


# Configuration
DATA_YAML = Path("/home/aanil/Data/aanil/side/yolo/datasets/keypoint_model/data.yaml")

MODEL_SIZE = "yolo26m"
MODEL_VARIANT = "pose"

EPOCHS = 500

# This is the network input size, not the original dataset image size.
# Variable-size source images are okay as long as labels are normalized per image.
IMG_SIZE = 640

DEVICE = [0, 1] if torch.cuda.is_available() else "cpu"

PROJECT_NAME = "/home/aanil/Data/aanil/side/yolo/outputs/yolo26_s_keypoint"
RUN_NAME = "new_keypointdata"

# Use rectangular batching for variable aspect ratios.
# This minimizes unnecessary padding inside each batch.
RECT = True

# Simple augmentation settings
AUGMENTATION = {
    # Color / lighting augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,

    # Geometric augmentation
    "degrees": 5.0,
    "translate": 0.10,
    "scale": 0.30,
    "shear": 2.0,
    "perspective": 0.0005,

    # Flips
    # Be careful with fliplr for keypoints:
    # if your pitch keypoints have left/right semantic meaning,
    # define flip_idx in data.yaml, otherwise disable fliplr.
    "fliplr": 0.5,
    "flipud": 0.0,

    # YOLO-style augmentation
    "mosaic": 0.5,
    "mixup": 0.05,
    "copy_paste": 0.0,

    # Turn off mosaic near the end for more stable final training
    "close_mosaic": 20,

    # Randomly resize batches around IMG_SIZE.
    # 0.25 means roughly 0.75x to 1.25x of IMG_SIZE.
    "multi_scale": 0.25,
}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(data_yaml: Path) -> dict:
    with data_yaml.open("r") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(path_value, dataset_root: Path) -> Path:
    """
    Resolve paths from data.yaml.
    Ultralytics data.yaml paths are often relative to the YAML parent or to `path:`.
    """
    path_value = Path(path_value)
    if path_value.is_absolute():
        return path_value
    return dataset_root / path_value


def get_image_files(image_dir: Path):
    if image_dir.is_file():
        with image_dir.open("r") as f:
            return [Path(line.strip()) for line in f if line.strip()]

    return sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )


def image_to_label_path(image_path: Path) -> Path:
    """
    Standard YOLO layout:
      images/train/foo.jpg -> labels/train/foo.txt
    """
    parts = list(image_path.parts)

    if "images" not in parts:
        raise ValueError(
            f"Cannot infer label path for {image_path}. "
            "Expected image path to contain an 'images' directory."
        )

    idx = parts.index("images")
    parts[idx] = "labels"

    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def validate_pose_labels_for_variable_images(data_yaml: Path):
    """
    Validates that labels are normalized and compatible with variable image sizes.

    This does not resize images or modify labels. It only catches common mistakes:
      - bbox/keypoint x/y values in pixel coordinates instead of normalized coordinates
      - malformed pose rows
      - missing labels
      - missing images
    """
    data = load_data_yaml(data_yaml)

    dataset_root = Path(data.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    # kpt_shape example: [N, 3]
    kpt_shape = data.get("kpt_shape", None)
    if not kpt_shape:
        raise ValueError(
            "data.yaml must define kpt_shape, e.g. kpt_shape: [4, 3] "
            "for 4 keypoints with x, y, visibility."
        )

    num_keypoints, keypoint_dims = int(kpt_shape[0]), int(kpt_shape[1])
    expected_cols = 5 + num_keypoints * keypoint_dims

    splits = ["train", "val"]
    errors = []

    for split in splits:
        if split not in data:
            continue

        image_dir = resolve_dataset_path(data[split], dataset_root)
        image_files = get_image_files(image_dir)

        if not image_files:
            errors.append(f"No images found for split '{split}' at {image_dir}")
            continue

        for image_path in image_files:
            if not image_path.is_absolute():
                image_path = dataset_root / image_path

            if not image_path.exists():
                errors.append(f"Missing image: {image_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                errors.append(f"Could not read image {image_path}: {e}")
                continue

            label_path = image_to_label_path(image_path)

            if not label_path.exists():
                # Empty-label images are allowed in detection, but for pitch-keypoint
                # training this is usually unintended. Change to warning if needed.
                errors.append(f"Missing label for image: {image_path}")
                continue

            with label_path.open("r") as f:
                rows = [line.strip() for line in f if line.strip()]

            for row_idx, row in enumerate(rows, start=1):
                values = row.split()

                if len(values) != expected_cols:
                    errors.append(
                        f"{label_path}:{row_idx} has {len(values)} columns, "
                        f"expected {expected_cols}. "
                        f"Check kpt_shape={kpt_shape} and label format."
                    )
                    continue

                try:
                    nums = [float(v) for v in values]
                except ValueError:
                    errors.append(f"{label_path}:{row_idx} contains non-numeric values.")
                    continue

                bbox = nums[1:5]
                keypoints = nums[5:]

                # YOLO bbox values must be normalized.
                if any(v < 0 or v > 1 for v in bbox):
                    errors.append(
                        f"{label_path}:{row_idx} bbox is not normalized. "
                        f"Image size is {img_w}x{img_h}, bbox={bbox}. "
                        "Use x/img_w, y/img_h, w/img_w, h/img_h."
                    )

                # Check keypoint x/y only, skip visibility.
                for k in range(num_keypoints):
                    base = k * keypoint_dims
                    kp_x = keypoints[base]
                    kp_y = keypoints[base + 1]

                    if kp_x < 0 or kp_x > 1 or kp_y < 0 or kp_y > 1:
                        errors.append(
                            f"{label_path}:{row_idx} keypoint {k} is not normalized. "
                            f"Image size is {img_w}x{img_h}, kp=({kp_x}, {kp_y}). "
                            "Use kp_x/img_w and kp_y/img_h."
                        )

    if errors:
        preview = "\n".join(errors[:30])
        raise RuntimeError(
            f"Dataset validation failed with {len(errors)} issue(s).\n\n"
            f"First issues:\n{preview}"
        )

    print("Dataset validation passed: labels appear normalized per image.")


def main():
    """Train YOLO26 pose model from scratch on variable-size images."""

    validate_pose_labels_for_variable_images(DATA_YAML)

    task = Task.init(
        project_name="PitchSense",
        task_name="yolo26m_keypoint_detection_baseline_aug_variable_imgsize",
        task_type=Task.TaskTypes.training,
    )

    print(f"Training YOLO26-{MODEL_SIZE}-{MODEL_VARIANT} from scratch")
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_YAML}")
    print(f"Training image size: {IMG_SIZE}")
    print(f"Rectangular batching: {RECT}")
    print("-" * 50)

    # For training from scratch, use the model YAML.
    # If you want pretrained weights instead, use f"{MODEL_SIZE}-{MODEL_VARIANT}.pt"
    # and set pretrained=True.
    model = YOLO(f"{MODEL_SIZE}-{MODEL_VARIANT}.yaml")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        rect=RECT,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        verbose=True,
        pretrained=False,
        patience=50,
        **AUGMENTATION,
    )

    print("-" * 50)
    print("Training complete!")
    print(f"Results saved to: {PROJECT_NAME}/{RUN_NAME}")

    task.close()


if __name__ == "__main__":
    main()