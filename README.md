# xView Object Detection Challenge 2018

This repository contains my work for the [xView Detection Challenge 2018](https://xviewdataset.org/). The primary goal is to detect a wide variety of objects in satellite imagery.

---

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

- Python 3.x
- PyTorch
- Other dependencies from `requirements.txt` (assuming one exists or should exist)

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/xView.git
    cd xView
    ```

2.  **Download the dataset:**
    Download the official xView dataset from the [official website](https://xviewdataset.org/). The dataset consists of:
    - Training set: 846 images
    - Validation set: 281 images

    The annotations are provided in GeoJSON format. You will also need the class labels mapping, which can be found in the [official xView data utilities repository](https://github.com/DIUx-xView/data_utilities/blob/master/xview_class_labels.txt). Download this file and place it in a relevant data folder.

3.  **Configure the project:**
    Edit the main YAML configuration file to set up your data paths, training settings, and hyperparameters.

4.  **Run the pipeline:**
    Execute the main script to start the data processing, model training, and inference pipeline.
    ```bash
    python run_all.py
    ```

5.  **Inference:**
    Once training is complete, run the inference script to evaluate the model, compute metrics, and visualize predictions.
    ```bash
    python inference.py
    ```

---

## 🛠️ Models & Architecture

This project explores different object detection models to find an optimal balance between accuracy, training/inference speed, and implementation complexity. All models were trained on an **NVIDIA RTX 4060 (8GB VRAM)**.

The primary model explored so far is:
- **[Faster R-CNN](https://arxiv.org/pdf/1506.01497)**

---

## 📊 Experiments & Results

Here's a summary of the experiments conducted and their results.

### Experiment 1: Baseline Faster R-CNN (2 Classes)
- **Description:** A baseline Faster R-CNN model was trained for 100 epochs on 2 classes ('Small Aircraft' and 'Passenger/Cargo Plane') with a batch size of 8.
- **Training Time:** ~30 seconds per epoch.
- **Key Finding:** Achieved a **mAP@0.5 of ~0.35**. Different learning rates were used for the backbone and the classification head. While modest, the results show promise.

### Experiment 2: Baseline Faster R-CNN (3 Classes)
- **Description:** The model was extended to 3 classes ('Small Aircraft', 'Passenger/Cargo Plane', 'Helicopter') and trained for 100 epochs with a batch size of 8.
- **Training Time:** ~33 seconds per epoch.
- **Key Finding:** The **mAP@0.5 dropped to ~0.3015**. Extending the training to 500 epochs did not yield improvements, suggesting the model was overfitting or required further tuning.

### Experiment 3: Faster R-CNN with Custom Anchors (3 Classes)
- **Description:** To better suit the scale of objects in satellite imagery, the default PyTorch anchor sizes were replaced with custom sizes: `((8,), (16,), (32,), (64,), (128,))`. The model was trained for 100 epochs on the same 3 classes.
- **Training Time:** ~33 seconds per epoch.
- **Key Finding:** The **mAP@0.5 improved to ~0.3148**, indicating that custom anchor sizes are beneficial for this dataset.

### Results Summary

Here you can see the results for each configuration

| Configuration | Number of Labels | mAP (IoU=0.50:0.95) | mAP50 (IoU=0.50 strict) | mAP75 (IoU=0.75 strict) | mAP (Small objects) | mAP (Medium objects) | mAP (Large objects) |
|:--------------|:-----------------|:--------------------|:------------------------|:------------------------|:--------------------|:---------------------|:--------------------|
| Exp 1         | 2                | 0.1699              | 0.3528                  | 0.1305                  | 0.1463              | 0.1827               | -1.0000             |
| Exp 2         | 3                | 0.1390              | 0.3015                  | 0.0967                  | 0.1188              | 0.2162               | -1.0000             |
| Exp 3         | 3                | 0.1505              | 0.3148                  | 0.1084                  | 0.1361              | 0.2276               | -1.0000             |

*Note: A value of -1.0 indicates that the metric was not applicable or no objects of that size were present.*

---

## 🔮 Future Work

Future improvements for this project include:
- [ ] Exploring other object detection models (e.g., YOLO, SSD).
- [ ] Converting the dataset to standard formats like COCO or YOLO for easier benchmarking.
- [ ] Improving the user experience for running experiments.
- [ ] Adding functionality to save and compare experiment plots.
- [ ] And much more!

Stay Tuned :)