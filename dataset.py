import os
import json
import random

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
import numpy as np

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
        """Retourne les transformations Albumentations pour des bbox en pixels."""
        if additional_transforms is None:
            additional_transforms = []

        transforms = [
            A.Resize(height=1024, width=1024),  # Redimensionne à 1024x1024
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
                format='pascal_voc',  # Utiliser Pascal VOC pour les pixels (x_min, y_min, x_max, y_max)
                label_fields=['class_labels'],
                min_visibility=0.0,
                min_area=0,
                clip=True
            ),
            keypoint_params=None
        )

    def get_image_info(self, idx):
        return self.coco_data['images'][idx]

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        img_info = self.coco_data['images'][idx]
        image_id = img_info['id']
        image_path = os.path.join(self.root_dir, 'images', img_info['file_name'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        annotations = [
            ann for ann in self.coco_data['annotations']
            if ann['image_id'] == image_id
        ]

        bboxes = []
        labels = []
        masks = []
        for ann in annotations:
            x_min, y_min, w, h = ann['bbox']
            x_max = x_min + w
            y_max = y_min + h

            if x_min < 0 or y_min < 0 or x_max > original_width or y_max > original_height:
                print(f"Bbox out of range for {img_info['file_name']}: {x_min}, {y_min}, {x_max}, {y_max} (image: {original_width}x{original_height})")
                continue

            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

            if 'segmentation' in ann:
                mask = utils.polygon_to_mask(ann['segmentation'], original_width, original_height)
                masks.append(mask)

        if masks:
            masks = np.stack(masks)
        else:
            masks = np.zeros((0, original_height, original_width), dtype=np.uint8)

        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels,
            masks=masks
        )

        image = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['class_labels']
        masks = transformed['masks']

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        target = {
            'boxes': bboxes,
            'labels': torch.as_tensor(labels, dtype=torch.float32),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([image_id]),
            'area': torch.as_tensor([ann['area'] for ann in annotations], dtype=torch.float32),
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

    val_dataset = SingleLabelCoco(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_val.json",
        is_train=False
    )

    test_dataset = SingleLabelCoco(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_test.json",
        is_train=False
    )

    for idx in range(len(train_dataset)):
        utils.visualize_augmented(train_dataset, idx=idx)


