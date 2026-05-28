import torch 
VAL_RATIO = 0.15
SEED = 10
EPOCHS = 200
IMGSZ = 1280
BATCH = 48
DEVICE = [0, 1] if torch.cuda.is_available() else "cpu"
