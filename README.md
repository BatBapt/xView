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

## Model used

- [Faster R-CNN](https://arxiv.org/pdf/1506.01497).

---

## Main files

You can download the repository and run the code, here is the good pipeline:

1. Download the dataset
2. Edit the YAML file to configure your path, settings and hyperparameters
3. Run the `run_all.py` to start the whole pipeline: data processing, training and inference
4. Once the training is complete, run the `inference.py` file to compute the metrics and visualize the predictions

---

## Experiments

What has been done so far ?

### 1. Baseline Faster R-CNN for 2 labels
I have trained a baseline model with a Faster R-CNN for 100 epochs for only **2** labels with a batch size of 8.
  - Trained on 2 labels: 'Small Aircraft' and 'Passenger/Cargo Plane'
  - Time / epochs ~= 30sec
  - **mAP (0.5) ~= 0.35**
  - Different learning rate were used for the backbone and the new head 
  - While the results are small, it also shows a good future.

### 2. Baseline Faster R-CNN for 3 labels

I have trained a baseline model with a Faster R-CNN for 100 epochs for only **3** labels with a batch size of 8.
- Trained on 3 labels: 'Small Aircraft' and 'Passenger/Cargo Plane' and 'Helicopter'
- Time / epochs ~= 33sec
- **mAP (0.5) ~= 0.3015**
- Different learning rate were used for the backbone and the new head 
- The results are smaller than the previous version with only 2 labels.
- This model was also trained with 500 epochs, but it didn't improve the model performance


### 3. Faster R-CNN for 3 labels with different anchors size
Instead of using the default Pytorch Anchors implementation in the model, which were designed for "normal" pictures,
I choose to use different anchors size: `((8,), (16,), (32,), (64,), (128,))`. The model was trained for only 100 epochs on **3** labels with a batch size of 8.
- Trained on 3 labels: 'Small Aircraft' and 'Passenger/Cargo Plane' and 'Helicopter'
- Time / epochs ~= 33sec
- **mAP (0.5) ~= 0.3148**
- Different learning rate were used for the backbone and the new head
- The results are better than the baseline version, suggesing
---

## Results

Here you can see the results for each configuration

| Configuration | Number of Labels | mAP (IoU=0.50:0.95) | mAP50 (IoU=0.50 strict) | mAP75 (IoU=0.75 strict) | mAP (Small objects) | mAP (Medium objects) | mAP (Large objects) |
|:--------------|:-----------------|:--------------------|:------------------------|:------------------------|:--------------------|:---------------------|:--------------------|
| Exp 1         | 2                | 0.1699              | 0.3528                  | 0.1305                  | 0.1463              | 0.1827               | -1.0000             |              |
| Exp 2         | 3                | 0.1390              | 0.3015                  | 0.0967                  | 0.1188              | 0.2162               | -1.0000             |
| Exp 3         | 3                | 0.1505              | 0.3148                  | 0.1084                  | 0.1361              | 0.2276               | -1.0000             |

--- 

## Future

Future improvements will arrive soon including:
- New models
- New data format (COCO or YOLO format)
- User-friendly experience to use the repo
- Save the plot
- And much more !

Stay Tuned :) 