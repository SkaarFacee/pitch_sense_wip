import torch 

SUBSET_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 10
EPOCHS = 200
IMGSZ = 1280
BATCH = 48
DEVICE = [0, 1] if torch.cuda.is_available() else "cpu"
CLEARML_DATASET_PROJECT = "PitchSense_v2"
CLEARML_DATASET_NAME = "Soccernet_ball_subset"
