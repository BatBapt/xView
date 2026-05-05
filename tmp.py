import os
import shutil
import geopandas as gpd
import random
import rasterio
import json

import configuration as cfg
import utils as utils


if __name__ == "__main__":
    geojson_label_file = os.path.join(cfg.LABELS_PATH, "xView_train.geojson")
    labels_map_file = os.path.join(cfg.LABELS_PATH, "xview_class_labels.txt")  # got from xview github

    gdf = gpd.read_file(geojson_label_file)
    labels_dict = utils.load_labels_txt(labels_map_file)

    target_labels = ["Small Aircraft"]
    folder_name = target_labels[0].replace(" ", "_")

    output_path = os.path.join(cfg.COCO_FORMAT_PATH, folder_name)

    output_images_path = os.path.join(output_path, "images")
    output_annotations_path = os.path.join(output_path, "annotations")
    output_json_path = os.path.join(output_annotations_path, f"{folder_name}_coco.json")

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_annotations_path, exist_ok=True)

    print(f"Output folders created at {output_path}")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": target_labels[0]}]
    }

    filtered_features = utils.filter_gdf_by_labels(gdf, labels_dict, target_labels)

    image_id_to_coco = {}
    coco_id = 1
    for _, row in filtered_features.iterrows():
        image_id = row["image_id"]
        if image_id not in image_id_to_coco:
            input_image_path = os.path.join(cfg.TRAIN_IMAGES_PATH, image_id)
            output_image_path = os.path.join(output_images_path, image_id)

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


        bbox = list(map(float, row["bounds_imcoords"].replace(" ", "").split(",")))
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        area = width*height

        polygon = row["geometry"]
        segmentation = [list(coord) for coord in polygon.exterior.coords]
        coco_data["annotations"].append({
            "id": len(coco_data["annotations"]) + 1,
            "image_id": image_id_to_coco[image_id],
            "category_id": 1,
            "bbox": [x_min, y_min, width, height],
            "segmentation": [segmentation],
            "area": area,
            "iscrowd": 0
        })


    with open(output_json_path, "w") as f:
        json.dump(coco_data, f)

    """
    # Display one sample

    filtered_features = utils.filter_gdf_by_labels(gdf, labels_dict, target_labels)
    list_image_id = filtered_features["image_id"].unique()
    image_id = random.choice(list_image_id)
    filtered_features = filtered_features[filtered_features["image_id"] == image_id]

    """






