import os
import torch
import yaml


# Torch settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_MODEL_PATH = "D:/models/torch/hub"  # Update path for your need, this is where pytorch models will be downloaded and stored
torch.hub.set_dir(TORCH_MODEL_PATH) # Un/comment this line to un/set the previous directory

# Load config
config_path = os.path.join(os.curdir, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Paths
MAIN_DATA_PATH = config["settings"]["data_path"]
TRAIN_IMAGES_PATH = os.path.join(MAIN_DATA_PATH, "train_images", "train_images")
LABELS_PATH = os.path.join(MAIN_DATA_PATH, "train_labels")
COCO_FORMAT_PATH = os.path.join(MAIN_DATA_PATH, "coco_format")
MODEL_WEIGHTS_PATH = os.path.join(os.curdir, "weights")

# Hyperparameters & Global settings
BATCH_SIZE = config["hyperparameters"]["batch_size"]
EPOCHS = config["hyperparameters"]["epochs"]
LR_BACKBONE = config["hyperparameters"]["lr_backbone"]
LR_HEAD = config["hyperparameters"]["lr_head"]

# Data and other
LABELS = config["settings"]["labels"]
NUM_WORKERS = config["settings"]["num_workers"]

if __name__ == "__main__":
    print(f"Main data path: {MAIN_DATA_PATH}")
    print(f"Traing images path: {TRAIN_IMAGES_PATH}")
    print(f"Labels path: {LABELS_PATH}")
    print(f"COCO format path: {COCO_FORMAT_PATH}")
    print(f"Model weights path: {MODEL_WEIGHTS_PATH}")
    print(f"Device: {DEVICE}")

    print("*"*50)

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate backbone: {LR_BACKBONE}")
    print(f"Learning rate head: {LR_HEAD}")

    print("*"*50)
    print(f"Current labels: {LABELS}")
    print(f"Number of label (including background): {len(LABELS) + 1}")
    print(f"Number of workers: {NUM_WORKERS}")