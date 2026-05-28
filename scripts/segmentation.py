from pathlib import Path

import torch
from ultralytics import YOLO
from clearml import Task


# Configuration
DATA_YAML = Path("/home/aanil/Data/aanil/side/yolo/datasets/segmentation/data.yaml")

MODEL_SIZE = "yolo26m"
MODEL_TASK = "seg"

EPOCHS = 200
IMG_SIZE = 640
PATIENCE = 50

DEVICE = [0, 1] if torch.cuda.is_available() else "cpu"

PROJECT_NAME = "/home/aanil/Data/aanil/side/yolo/outputs/yolo26_segmentation"
RUN_NAME = "segmentation_baseline"

CLEARML_PROJECT_NAME = "PitchSense_v2"
CLEARML_TASK_NAME = "yolo26m_segmentation_baseline"

RECT = True


def main():
    """Train a YOLO segmentation model with ClearML tracking."""

    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name=CLEARML_TASK_NAME,
        task_type=Task.TaskTypes.training,
    )

    config = {
        "data_yaml": str(DATA_YAML),
        "model_size": MODEL_SIZE,
        "model_task": MODEL_TASK,
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "patience": PATIENCE,
        "device": DEVICE,
        "project_name": PROJECT_NAME,
        "run_name": RUN_NAME,
        "rect": RECT,
        "pretrained": False,
    }

    task.connect(config)

    print(f"Training {MODEL_SIZE}-{MODEL_TASK}")
    print(f"Device: {DEVICE}")
    print(f"Data YAML: {DATA_YAML}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Rectangular batching: {RECT}")
    print("-" * 50)

    model = YOLO(f"{MODEL_SIZE}-{MODEL_TASK}.yaml")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        rect=RECT,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        patience=PATIENCE,
        pretrained=False,
        verbose=True,
    )

    print("-" * 50)
    print("Training complete!")
    print(f"Results saved to: {PROJECT_NAME}/{RUN_NAME}")

    task.close()

    return results


if __name__ == "__main__":
    main()