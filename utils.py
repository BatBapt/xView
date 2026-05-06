import os
import shutil
import rasterio
import json
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

import configuration as cfg


def load_labels_txt(labels_file):
    labels_map = {}
    with open(labels_file, 'r') as f:
        for line in f:
            class_id, class_name = line.split(":")
            labels_map[int(class_id)] = class_name.strip()
    return labels_map


def plot_image(image_id, filtered_features, class_name):
    image_path = os.path.join(cfg.TRAIN_IMAGES_PATH, image_id)
    with rasterio.open(image_path) as src:
        img = src.read()

        if img.shape[0] == 1:
            img_rgb = img[0]
        elif img.shape[0] >= 3:
            img_rgb = np.dstack((img[0], img[1], img[2]))
        else:
            raise ValueError(f"Invalid image [{image_id}] shape: {img.shape[0]}")

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_rgb)

        patches = []
        for _, row in filtered_features.iterrows():
            x_min, y_min, x_max, y_max = map(float, row["bounds_imcoords"].replace(" ", "").split(","))
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            patches.append(rect)

            ax.text(
                x_min, y_min - 5,
                class_name,
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.7, edgecolor="none", pad=1)
            )

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)
        ax.set_title(f"Annotations for {image_id}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def filter_gdf_by_labels(gdf, labels_dict, target_labels):
    target_type_ids = [
        type_id for type_id, class_name in labels_dict.items()
        if class_name in target_labels
    ]
    return gdf[gdf["type_id"].isin(target_type_ids)]


def visualize_augmented(dataset, idx=0, class_colors=None):
    img, target = dataset[idx]

    for key, value in target.items():
        print(f"{key}: {value} | {value.shape}")

    img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    img_info = dataset.get_image_info(idx)
    image_id = img_info['id']

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)

    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()

    class_names = [dataset.category_id_to_name[int(label)] for label in labels]

    if class_colors is None:
        class_colors = {
            class_name: plt.cm.tab10(i % 10)[:3]
            for i, class_name in enumerate(set(class_names))
        }

    for i, (box, label, class_name) in enumerate(zip(boxes, labels, class_names)):
        x_min, y_min, x_max, y_max = box
        color = class_colors.get(class_name, "red")

        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x_min, y_min - 5,
            f"{class_name}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1)
        )

    ax.set_title(f"Image ID: {image_id} | Annotations: {len(boxes)}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def prepare_xview2coco(target_labels, filtered_features, labels_dict, output_path, folder_name, val_ratio=0.2, random_seed=42):

    output_images_path = os.path.join(output_path, "images")
    output_annotations_path = os.path.join(output_path, "annotations")

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_annotations_path, exist_ok=True)

    unique_image_ids = filtered_features["image_id"].unique().tolist()
    random.Random(random_seed).shuffle(unique_image_ids)

    val_size = int(len(unique_image_ids) * val_ratio)

    val_image_ids = unique_image_ids[:val_size]
    train_image_ids = unique_image_ids[val_size:]

    category_mapping = {idx + 1: label for idx, label in enumerate(target_labels)}
    name_to_category_id = {name: cat_id for cat_id, name in category_mapping.items()}

    def generate_coco_json(image_ids, split_name):
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": cat_id, "name": name} for cat_id, name in category_mapping.items()]
        }

        image_id_to_coco = {}
        coco_id = 1
        annotation_id = 1

        for image_id in image_ids:
            input_image_path = os.path.join(cfg.TRAIN_IMAGES_PATH, image_id)
            output_image_path = os.path.join(output_images_path, image_id)

            if not os.path.exists(input_image_path):
                print(f"Skipping image {input_image_path} -> does not exists")
                continue

            if not os.path.exists(output_image_path):
                shutil.copy(input_image_path, output_image_path)

            with rasterio.open(input_image_path) as src:
                width, height = src.width, src.height

            image_id_to_coco[image_id] = coco_id
            coco_data["images"].append({
                "id": coco_id,
                "file_name": image_id,
                "width": width,
                "height": height
            })
            coco_id += 1

        for _, row in filtered_features.iterrows():
            if row["image_id"] in image_id_to_coco:
                bbox = list(map(float, row["bounds_imcoords"].replace(" ", "").split(",")))
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                area = width * height

                original_type_id = row["type_id"]
                class_name = labels_dict[original_type_id]
                category_id = name_to_category_id[class_name]

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id_to_coco[row["image_id"]],
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1

        output_json_path = os.path.join(output_annotations_path, f"{folder_name}_{split_name}.json")
        with open(output_json_path, "w") as f:
            json.dump(coco_data, f)

        return coco_data

    generate_coco_json(train_image_ids, "train")
    generate_coco_json(val_image_ids, "val")

    print(f"COCO dataset created at {output_path} with splits: train={len(train_image_ids)}, val={len(val_image_ids)}")


def collate_fn(batch):
    return tuple(zip(*batch))