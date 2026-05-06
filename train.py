import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as utils
import dataset as my_dataset
import models as my_models
import configuration as cfg


def get_dataloaders(root_dir, folder_name, batch_size=4, num_workers=2):

    print("Dataset initialization")

    train_dataset = my_dataset.XViewCocoDataset(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_train.json",
        is_train=True
    )

    val_dataset = my_dataset.XViewCocoDataset(
        root_dir=root_dir,
        annotation_file=f"annotations/{folder_name}_val.json",
        is_train=False
    )

    print(f"Number of element in the training set: {len(train_dataset)}")
    print(f"Number of element in the validation set: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    total_loss = 0.0

    loop = tqdm(data_loader, desc=f"Epoch [{epoch}]", leave=True)

    for images, targets in loop:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(data_loader)
    print(f"\nMean loss at epoch {epoch} : {avg_loss:.4f}")

    return avg_loss


def validate(model, data_loader, device, epoch):
    model.train()  # trick from Faster RCNN
    total_loss = 0.0

    loop = tqdm(data_loader, desc=f"Epoch [{epoch}] Valid", leave=True, colour='green')

    with torch.no_grad():
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            loop.set_postfix(val_loss=losses.item())

    return total_loss / len(data_loader)


def train(target_labels, folder_name, root_dir, weight_dir, weights_name=None, device="cpu"):
    if weights_name is None:
        weights_name = {"best": "faster_rcnn_best.pth", "last": "faster_rcnn_last.pth"}

    weights_last_path = os.path.join(weight_dir, weights_name["last"])
    weights_best_path = os.path.join(weight_dir, weights_name["best"])

    num_classes = len(target_labels) + 1  # +1 bc of background
    num_epochs = cfg.EPOCHS
    batch_size = cfg.BATCH_SIZE
    num_workers = cfg.NUM_WORKERS

    base_lr = cfg.LR_HEAD  # new head
    backbone_lr = cfg.LR_BACKBONE # backbone

    train_loader, val_loader = get_dataloaders(root_dir, folder_name, batch_size=batch_size, num_workers=num_workers)

    print("Loading model")
    model = my_models.get_model_instance_segmentation(num_classes)
    model.to(device)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'roi_heads' in name or 'rpn' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': base_lr}
    ], momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')

    print("Starting training !")
    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        val_loss = validate(model, val_loader, device, epoch)

        lr_scheduler.step()

        print(f"\nResult epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), weights_last_path)

        if val_loss < best_val_loss:
            print(f"Model saved with validation loss: {val_loss:.4f}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_best_path)


if __name__ == "__main__":
    print("Hello, I'm the training script")