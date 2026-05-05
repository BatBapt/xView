import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_MODEL_PATH = "D:/models/torch/hub"  # Update path for your need, this is where pytorch models will be downloaded and stored
torch.hub.set_dir(TORCH_MODEL_PATH) # Un/comment this line to un/set the previous directory

MAIN_DATA_PATH = "D:/Programmation/IA/datas/xView"

TRAIN_IMAGES_PATH = os.path.join(MAIN_DATA_PATH, "train_images", "train_images")
LABELS_PATH = os.path.join(MAIN_DATA_PATH, "train_labels")
COCO_FORMAT_PATH = os.path.join(MAIN_DATA_PATH, "coco_format")