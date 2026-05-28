import pathlib
import shutil

import torch
from clearml import Task, OutputModel, Dataset
from ultralytics import YOLO
from constants import VAL_RATIO,SEED,EPOCHS,IMGSZ,BATCH,DEVICE,CLEARML_DATASET_PROJECT,CLEARML_DATASET_NAME

MODEL_NAME = "yolo26n.yaml"
OUTPUT_ROOT = pathlib.Path("/home/aanil/Data/aanil/side/yolo/outputs/26n_ball_pretrained")
SAVE_DIR = OUTPUT_ROOT / "saved_models"
SUBSET_RATIO=1.0

CLEARML_DATASET_ID = '300ca95787984633bc04cf2155a4c968'  # set this to a specific dataset ID if you want to pin the exact dataset


def pull_yolo_dataset_from_clearml() -> tuple[Dataset, pathlib.Path]:
    if CLEARML_DATASET_ID is not None:
        clearml_dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    else:
        clearml_dataset = Dataset.get(
            dataset_project=CLEARML_DATASET_PROJECT,
            dataset_name=CLEARML_DATASET_NAME,
        )

    local_dataset_path = pathlib.Path(clearml_dataset.get_local_copy())
    yaml_path = local_dataset_path / "dataset.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found in ClearML dataset copy: {yaml_path}")

    print("ClearML dataset pulled.")
    print(f"ClearML dataset ID: {clearml_dataset.id}")
    print(f"Local dataset path: {local_dataset_path}")
    print(f"YOLO dataset yaml: {yaml_path}")

    return clearml_dataset, yaml_path


def copy_best_weights(save_dir: pathlib.Path, run_name: str) -> pathlib.Path | None:
    best_weights = save_dir / run_name / "weights" / "best.pt"
    final_model_path = save_dir / "yolo26n_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, final_model_path)
        print(f"Saved best model to: {final_model_path}")
        return final_model_path

    print("[WARN] best.pt not found after training.")
    return None


def main() -> None:
    task = Task.init(
        project_name="PitchSense_v2",
        task_name="yolo26n_baseline",
        task_type=Task.TaskTypes.training,
    )

    task.connect(
        {
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
            "clearml_dataset_project": CLEARML_DATASET_PROJECT,
            "clearml_dataset_name": CLEARML_DATASET_NAME,
            "clearml_dataset_id": CLEARML_DATASET_ID,
        }
    )

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Training device:", DEVICE)

    clearml_dataset, yaml_path = pull_yolo_dataset_from_clearml()

    task.upload_artifact("dataset_yaml", artifact_object=str(yaml_path))
    task.upload_artifact("clearml_dataset_id", artifact_object=clearml_dataset.id)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    run_name = "yolo26n_baseline"

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(SAVE_DIR),
        name=run_name,
        pretrained=True,
        verbose=True,
    )

    final_model_path = copy_best_weights(SAVE_DIR, run_name)

    if final_model_path and final_model_path.exists():
        output_model = OutputModel(task=task, name="yolo26n_best")
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
            name="test_eval",
            verbose=True,
        )

        # task.upload_artifact("test_results", artifact_object=str(test_results))

    task.upload_artifact("output_root", artifact_object=str(OUTPUT_ROOT))
    print("Training complete.")
    task.close()


if __name__ == "__main__":
    main()