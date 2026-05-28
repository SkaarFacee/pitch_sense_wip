import pathlib
import shutil
from typing import Dict, List, Tuple

import torch
import yaml
from clearml import Task, OutputModel, Dataset
from torch.utils.data import random_split, Subset
from ultralytics import YOLO
from constants import VAL_RATIO,SEED,EPOCHS,IMGSZ,BATCH,DEVICE
from datasetLoader import PitchSenseDataset

MODEL_NAME = "yolo26n.yaml"
BASE_PATH = "/home/aanil/Data/aanil/side/yolo/datasets/Soccernet/tracking"
OUTPUT_ROOT = pathlib.Path("/home/aanil/Data/aanil/side/yolo/outputs/yolo_26n_ball_200epochs")
SAVE_DIR = OUTPUT_ROOT / "saved_models"


SUBSET_RATIO = 1.0

# Train only this class.
TARGET_CLASSES = ["ball"]

# True:
#   Keep images that do not contain a ball as negative/background examples.
#
# False:
#   Export only images that contain at least one ball.
#
# I recommend starting with True for ball detection.
KEEP_IMAGES_WITHOUT_TARGET = True


class PATHS:
    train_path = pathlib.Path(f"{BASE_PATH}/train")
    test_path = pathlib.Path(f"{BASE_PATH}/test")


def yolo_box_from_xywh(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    box_w = w / img_w
    box_h = h / img_h
    return x_center, y_center, box_w, box_h


def clamp_box(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
) -> Tuple[float, float, float, float]:
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-6), 1.0)
    bh = min(max(bh, 1e-6), 1.0)
    return cx, cy, bw, bh


def get_image_size(sample: dict) -> Tuple[int, int]:
    config = sample["config"]
    img_w = config.get("imwidth")
    img_h = config.get("imheight")

    if img_w is None or img_h is None:
        raise ValueError(f"Missing image size in config for sample: {sample['img_path']}")

    return int(img_w), int(img_h)


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower()


def build_class_mapping(
    target_classes: List[str],
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a YOLO class mapping using only the target classes.

    For TARGET_CLASSES = ["ball"], this returns:
        class_map = {"ball": 0}
        class_names = ["ball"]
    """
    class_names = [normalize_class_name(c) for c in target_classes]

    if not class_names:
        raise ValueError("TARGET_CLASSES cannot be empty.")

    class_map = {name: idx for idx, name in enumerate(class_names)}
    return class_map, class_names


def make_unique_stem(img_path: pathlib.Path) -> str:
    match_name = img_path.parent.parent.name
    return f"{match_name}_{img_path.stem}"


def create_subset(dataset, subset_ratio: float, seed: int):
    subset_size = int(len(dataset) * subset_ratio)

    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)


def export_split(
    samples,
    split_name: str,
    out_root: pathlib.Path,
    class_map: Dict[str, int],
    keep_images_without_target: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Export a YOLO split containing only the classes in class_map.

    For ball-only training:
        class_map = {"ball": 0}

    Any non-ball objects are ignored.
    """
    images_dir = out_root / "images" / split_name
    labels_dir = out_root / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0
    written_objects = 0
    skipped_images = 0

    for sample in samples:
        img_path = pathlib.Path(sample["img_path"])
        gt_df = sample["gt"]

        if not img_path.exists():
            print(f"[WARN] Missing image, skipping: {img_path}")
            continue

        img_w, img_h = get_image_size(sample)
        stem = make_unique_stem(img_path)

        label_lines = []

        if gt_df is not None and not gt_df.empty:
            for _, row in gt_df.iterrows():
                raw_name = row.get("name", None)
                if raw_name is None:
                    continue

                class_name = normalize_class_name(raw_name)

                # Keep only the target class, for example "ball".
                if class_name not in class_map:
                    continue

                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])

                if w <= 0 or h <= 0:
                    continue

                # Since class_map = {"ball": 0}, all exported objects are class 0.
                class_id = class_map[class_name]

                cx, cy, bw, bh = yolo_box_from_xywh(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    img_w=img_w,
                    img_h=img_h,
                )
                cx, cy, bw, bh = clamp_box(cx, cy, bw, bh)

                label_lines.append(
                    f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
                written_objects += 1

        # If the image has no ball labels, either keep it as a negative image
        # with an empty .txt file, or skip it entirely.
        if not label_lines and not keep_images_without_target:
            skipped_images += 1
            continue

        dst_img_path = images_dir / f"{stem}{img_path.suffix}"
        dst_label_path = labels_dir / f"{stem}.txt"

        shutil.copy2(img_path, dst_img_path)
        written_images += 1

        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        written_labels += 1

    print(
        f"[{split_name}] exported images: {written_images}, "
        f"labels: {written_labels}, "
        f"target objects: {written_objects}, "
        f"skipped images: {skipped_images}"
    )

    return written_images, written_labels, written_objects, skipped_images


def write_dataset_yaml(out_root: pathlib.Path, class_names: List[str]) -> pathlib.Path:
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }

    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return yaml_path


def copy_best_weights(save_dir: pathlib.Path, run_name: str) -> pathlib.Path | None:
    best_weights = save_dir / run_name / "weights" / "best.pt"
    final_model_path = save_dir / "yolo26n_ball_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, final_model_path)
        print(f"Saved best model to: {final_model_path}")
        return final_model_path

    print("[WARN] best.pt not found after training.")
    return None


def push_yolo_dataset_to_clearml(out_root: pathlib.Path) -> Dataset:
    clearml_dataset = Dataset.create(
        dataset_name="Soccernet_ball_subset",
        dataset_project="PitchSense_v2",
        description=(
            "YOLO-format SoccerNet tracking export for ball-only detection. "
            "Contains images/train, images/val, images/test, "
            "labels/train, labels/val, labels/test, and dataset.yaml. "
            "Only the ball class is exported as class id 0."
        ),
    )

    clearml_dataset.add_files(path=str(out_root))
    clearml_dataset.upload()
    clearml_dataset.finalize()

    print("ClearML dataset uploaded.")
    print(f"ClearML dataset ID: {clearml_dataset.id}")

    return clearml_dataset


def main() -> None:
    task = Task.init(
        project_name="PitchSense_v2",
        task_name="yolo26n_ball_only",
        task_type=Task.TaskTypes.training,
    )

    task.connect(
        {
            "base_path": BASE_PATH,
            "output_root": str(OUTPUT_ROOT),
            "save_dir": str(SAVE_DIR),
            "model_name": MODEL_NAME,
            "subset_ratio": SUBSET_RATIO,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "target_classes": TARGET_CLASSES,
            "keep_images_without_target": KEEP_IMAGES_WITHOUT_TARGET,
            "train_path": str(PATHS.train_path),
            "test_path": str(PATHS.test_path),
        }
    )

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Training device:", DEVICE)

    train_root = PATHS.train_path
    test_root = PATHS.test_path

    full_dataset = PitchSenseDataset([train_root])
    full_test_dataset = PitchSenseDataset([test_root])

    dataset = create_subset(full_dataset, SUBSET_RATIO, SEED)
    test_dataset = create_subset(full_test_dataset, SUBSET_RATIO, SEED)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"Full dataset samples: {len(full_dataset)}")
    print(f"Subset dataset samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Full test samples: {len(full_test_dataset)}")
    print(f"Subset test samples: {len(test_dataset)}")

    class_map, class_names = build_class_mapping(TARGET_CLASSES)
    print("Class mapping:", class_map)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_stats = export_split(
        train_dataset,
        "train",
        OUTPUT_ROOT,
        class_map,
        keep_images_without_target=KEEP_IMAGES_WITHOUT_TARGET,
    )

    val_stats = export_split(
        val_dataset,
        "val",
        OUTPUT_ROOT,
        class_map,
        keep_images_without_target=KEEP_IMAGES_WITHOUT_TARGET,
    )

    test_stats = export_split(
        test_dataset,
        "test",
        OUTPUT_ROOT,
        class_map,
        keep_images_without_target=KEEP_IMAGES_WITHOUT_TARGET,
    )

    total_target_objects = train_stats[2] + val_stats[2] + test_stats[2]

    if total_target_objects == 0:
        raise RuntimeError(
            "No target objects were exported. "
            "Check that the dataset class name is exactly 'ball' after normalization. "
            "You may need to inspect gt_df['name'].unique()."
        )

    print(f"Total exported target objects: {total_target_objects}")

    yaml_path = write_dataset_yaml(OUTPUT_ROOT, class_names)
    print(f"YOLO dataset yaml written to: {yaml_path}")

    task.upload_artifact("dataset_yaml", artifact_object=str(yaml_path))

    clearml_dataset = push_yolo_dataset_to_clearml(OUTPUT_ROOT)
    task.upload_artifact("clearml_dataset_id", artifact_object=clearml_dataset.id)

    run_name = "yolo26n_ball_only"

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(SAVE_DIR),
        name=run_name,
        pretrained=False,
        verbose=True,
    )

    final_model_path = copy_best_weights(SAVE_DIR, run_name)

    if final_model_path and final_model_path.exists():
        output_model = OutputModel(task=task, name="yolo26n_ball_best")
        output_model.update_weights(weights_filename=str(final_model_path))

        task.upload_artifact("best_model", artifact_object=str(final_model_path))

        model = YOLO(str(final_model_path))

        test_results = model.val(
            data=str(yaml_path),
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            project=str(SAVE_DIR),
            name="test_eval_ball_only",
            verbose=True,
        )

        task.upload_artifact("test_results", artifact_object=str(test_results))

    task.upload_artifact("output_root", artifact_object=str(OUTPUT_ROOT))

    print("Training complete.")
    task.close()


if __name__ == "__main__":
    main()