# xView Detection Challenge

All this repository is about the xView Detection Challenge in 2018.
The goal is to detect a large range of object within a satellite image

More information [here](https://xviewdataset.org/)

---

## Data

The dataset includes 2 sets:
- A training set with 846 images
- A validation set with 281 images

The annotation are in GEOJson format.

But you can also use the mapping `ID:CLASS_NAME` file in the official [repository of xView](https://github.com/DIUx-xView/data_utilities/blob/master/xview_class_labels.txt)
Just download it and place in your folder

---

## Global

This repository will contain different modules based on which choice I make during exploration and implementation.
The goal is to use different models to see which one is the best, with a certain trade-off between accuracy / training time / inference time / implementation complexity

All the models will be trained with my GPU, Nvidia RTX 4060 (8GB VRAM)

---

## Main files

You can download the repository and run the code, here is the good pipeline:

1. Download the dataset
2. Edit the YAML file to configure your path, settings and hyperparameters
3. Run the script `run_all.py` to convert the xView format into COCO annotations
4. Run the script `dataset.py` to validate the conversation and visualize samples
5. Run the script `train.py` to train the model available in `models.py`
6. Once the training is complete, run the `inference.py` file to compute the metrics and visualize the predictions

---

## Experiments

What has been done so far ?

### Baseline Faster R-CNN for 2 labels
- I have trained a baseline model with [Faster R-CNN](https://arxiv.org/pdf/1506.01497) for 100 epochs for only 2 labels with a batch size of 8.
  - Trained on 2 labels: 'Small Aircraft' and 'Passenger/Cargo Plane'
  - Time / epochs ~= 30sec
  - **mAP (0.5) ~= 0.35**
  - Different learning rate were used for the backbone and the new head 
  - While the results are small, it also shows a good future.
  
| Metric                  | Value   |
|:------------------------|:--------|
| mAP (IoU=0.50:0.95)     | 0.1699  |
| mAP50 (IoU=0.50 strict) | 0.3528  |
| mAP75 (IoU=0.75 strict) | 0.1305  |
| mAP (Small objects)     | 0.1463  |
| mAP (Medium objects)    | 0.1827  |
| mAP (Large objects)     | -1.0000 |


### Baseline Faster R-CNN for 3 labels

**IN PROGESS**

---

## Future

Future improvements will arrive soon including:
- New models
- New data format (COCO or YOLO format)
- User-friendly experience to use the repo
- Save the plot
- And much more !

Stay Tuned :) 