import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import configuration as cfg
import utils as utils
import dataset as my_dataset


def plot_predictions(image_rgb, dataset, predictions, target, confidence_threshold=0.5):

    img_display = np.clip(image_rgb, 0, 1)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_display)

    gt_boxes = target['boxes'].cpu().numpy()
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = Rectangle((x_min, y_min), width, height,
                         linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, "True object", color="red", fontsize=8, fontweight='bold')

    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()

    count = 0
    for box, score, label_id in zip(pred_boxes, pred_scores, pred_labels):
        if score >= confidence_threshold:
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min

            class_name = dataset.category_id_to_name[label_id]

            rect = Rectangle((x_min, y_min), width, height,
                             linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

            text = f"{class_name}: {score:.2f}"
            ax.text(x_min, y_max + 15, text, color="black", fontsize=10, fontweight='bold',
                    bbox=dict(facecolor="lime", alpha=0.8, edgecolor="none", pad=2))
            count += 1

    image_id = target['image_id'].item()
    ax.set_title(
        f"Image ID: {image_id} | True Object: {len(gt_boxes)} | Detected: {count} (Threshold: {confidence_threshold})")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Ground Truth'),
        Line2D([0], [0], color='lime', lw=2, label='Model Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def evaluate_and_infer(folder_name, root_dir, model, threshold=0.5, visualize=False, device="cpu"):
    if not os.path.exists(root_dir):
        print(f"Dataset directory not found at {root_dir}")
        return None

    print("Loading test set...")
    test_dataset = my_dataset.XViewCocoDataset(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_val.json",
        is_train=False
    )
    print(f"{len(test_dataset)} images found.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    metric = MeanAveragePrecision(iou_type="bbox")

    print("Loading model weights and starting evaluation...")
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            print(f"Processing batch {i + 1}/{len(test_loader)}...")

            images_device = list(img.to(device) for img in images)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images_device)

            metric.update(preds, targets_device)

            if visualize:
                for img_tensor, pred, target in zip(images, preds, targets):
                    image_for_plot = img_tensor.permute(1, 2, 0).cpu().numpy()

                    plot_predictions(image_for_plot, test_dataset, pred, target, threshold)

    results = metric.compute()

    print("\n" + "*" * 50)
    print("Evaluation results")
    print("*" * 50)
    print(f"mAP (IoU=0.50:0.95)      : {results['map'].item():.4f}")
    print(f"mAP50 (IoU=0.50 strict)  : {results['map_50'].item():.4f}")
    print(f"mAP75 (IoU=0.75 strict)  : {results['map_75'].item():.4f}")
    print(f"mAP (Small objects)      : {results['map_small'].item():.4f}")
    print(f"mAP (Medium objects)     : {results['map_medium'].item():.4f}")
    print(f"mAP (Large objects)      : {results['map_large'].item():.4f}")
    print("*" * 50 + "\n")

    return results


if __name__ == "__main__":
    print("Hello, I'm the inference script")