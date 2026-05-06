import os
import geopandas as gpd

import configuration as cfg
import utils as utils


if __name__ == "__main__":
    geojson_label_file = os.path.join(cfg.LABELS_PATH, "xView_train.geojson")
    labels_map_file = os.path.join(cfg.LABELS_PATH, "xview_class_labels.txt")  # got from xview github

    gdf = gpd.read_file(geojson_label_file)
    labels_dict = utils.load_labels_txt(labels_map_file)

    target_labels = ["Small Aircraft", "Passenger/Cargo Plane"]

    filtered_features = utils.filter_gdf_by_labels(gdf, labels_dict, target_labels)

    utils.prepare_xview2coco(
        target_labels,
        filtered_features,
        labels_dict,
        val_ratio=0.2,
        random_seed=42
    )


    """
    # Display one sample

    filtered_features = utils.filter_gdf_by_labels(gdf, labels_dict, target_labels)
    list_image_id = filtered_features["image_id"].unique()
    image_id = random.choice(list_image_id)
    filtered_features = filtered_features[filtered_features["image_id"] == image_id]

    """






