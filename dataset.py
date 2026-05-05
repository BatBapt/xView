import os
import json
import random

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import rasterio

import utils as utils
import configuration as cfg


class SingleLabelCoco(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, is_train=True):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.is_train = is_train

        annotation_file_full_path = os.path.join(root_dir, annotation_file)
        if not os.path.exists(annotation_file_full_path):
            print(f"Error: {annotation_file_full_path} does not exists")
            exit()

        with open(annotation_file_full_path, "r") as f:
            self.coco_data = json.load(f)

        self.images_ids = [img["id"] for img in self.coco_data["images"]]
        self.image_id_to_index = {
            img["id"]: idx for idx, img in enumerate(self.coco_data["images"])
        }

        self.category_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        self.transform = self._get_transforms(transform)

    def _get_transforms(self, additional_transforms=None):
        if additional_transforms is None:
            additional_transforms = []

        transforms = [
            A.Resize(height=1024, width=1024),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ]

        if self.is_train:
            train_transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2, brightness_limit=0.2, contrast_limit=0.2),
            ]
            transforms = train_transforms + transforms

        if additional_transforms:
            transforms.extend(additional_transforms)

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.0,
                min_area=0,
                clip=True
            )
        )

    def get_image_info(self, idx):
        return self.coco_data['images'][idx]

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        img_info = self.coco_data['images'][idx]
        image_id = img_info['id']
        image_path = os.path.join(self.root_dir, 'images', img_info['file_name'])

        # Utilisation de rasterio pour lire correctement les .tif
        with rasterio.open(image_path) as src:
            image = src.read()

        if image.shape[0] >= 3:
            image = np.transpose(image[:3], (1, 2, 0))
        elif image.shape[0] == 1:
            image = np.stack((image[0], image[0], image[0]), axis=-1)

        if image.dtype != np.uint8:
            image = (image.astype(np.float32) / np.max(image) * 255).astype(np.uint8)

        original_height, original_width = image.shape[:2]

        annotations = [
            ann for ann in self.coco_data['annotations']
            if ann['image_id'] == image_id
        ]

        bboxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, w, h = ann['bbox']
            x_max = x_min + w
            y_max = y_min + h

            if x_min < 0 or y_min < 0 or x_max > original_width or y_max > original_height:
                # On ignore silencieusement ou on print si besoin
                continue

            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        # Application de l'augmentation uniquement sur l'image et les bboxes
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )

        image = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['class_labels']

        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.float32)
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        target = {
            'boxes': bboxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': area,
            'iscrowd': torch.zeros((len(bboxes),), dtype=torch.int64)
        }

        return image, target


if __name__ == "__main__":
    target_labels = ["Small Aircraft"]
    folder_name = target_labels[0].replace(" ", "_")
    root_dir = os.path.join(cfg.COCO_FORMAT_PATH, folder_name)

    train_dataset = SingleLabelCoco(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_train.json",
        is_train=True
    )

    idx = np.random.randint(0, len(train_dataset))
    utils.visualize_augmented(train_dataset, idx=idx)
