import os
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
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
        transform = src.transform  # Pour convertir lon/lat → pixels

        # Gestion des bandes
        if img.shape[0] == 1:
            img_rgb = img[0]
        elif img.shape[0] >= 3:
            img_rgb = np.dstack((img[0], img[1], img[2]))
        else:
            raise ValueError(f"Nombre de bandes non supporté: {img.shape[0]}")

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

        ax.set_title(f"Annotations pour {image_id}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def filter_gdf_by_labels(gdf, labels_dict, target_labels):
    target_type_ids = [
        type_id for type_id, class_name in labels_dict.items()
        if class_name in target_labels
    ]

    filtered_features = gdf[gdf["type_id"].isin(target_type_ids)]

    return filtered_features