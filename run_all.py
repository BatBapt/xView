import os
import geopandas as gpd
import random
import numpy as np

import configuration as cfg
import utils as utils
import dataset as my_dataset
import train as train_model
import inference as inference_model


if __name__ == "__main__":
    target_labels = cfg.LABELS
    folder_name = "_".join([label.replace(" ", "_").replace("/", "_") for label in target_labels])
    root_dir = os.path.join(cfg.COCO_FORMAT_PATH, folder_name)

    print(f"Running pipeline for labels: {target_labels}")

    model_name = f"baseline_faster_rcnn_{len(target_labels)}_labels"
    weights_dir = os.path.join(cfg.MODEL_WEIGHTS_PATH, model_name)
    os.makedirs(weights_dir, exist_ok=True)

    visualize = True

    if not os.path.exists(root_dir):
        geojson_label_file = os.path.join(cfg.LABELS_PATH, "xView_train.geojson")
        labels_map_file = os.path.join(cfg.LABELS_PATH, "xview_class_labels.txt")  # got from xview github

        gdf = gpd.read_file(geojson_label_file)
        labels_dict = utils.load_labels_txt(labels_map_file)

        filtered_features = utils.filter_gdf_by_labels(gdf, labels_dict, target_labels)

        utils.prepare_xview2coco(
            target_labels,
            filtered_features,
            labels_dict,
            output_path=root_dir,
            folder_name=folder_name,
            val_ratio=0.2,
            random_seed=42
        )

        if visualize:
            filtered_features = utils.filter_gdf_by_labels(gdf, labels_dict, target_labels)
            list_image_id = filtered_features["image_id"].unique()
            image_id = random.choice(list_image_id)
            filtered_features = filtered_features[filtered_features["image_id"] == image_id]

    if not os.path.exists(root_dir):
        print(f"Dataset directory not found at {root_dir}")
        exit()

    train_dataset = my_dataset.XViewCocoDataset(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_train.json",
        is_train=True
    )

    if visualize:
        idx = np.random.randint(0, len(train_dataset))
        utils.visualize_augmented(train_dataset, idx=idx)

    weights_best_path = train_model.train(
        target_labels,
        folder_name,
        root_dir,
        weights_dir,
        device=cfg.DEVICE
    )

    print("Training done !")

    inference_model.inference(
        target_labels,
        folder_name,
        root_dir,
        weights_best_path,
        device=cfg.DEVICE
    )





