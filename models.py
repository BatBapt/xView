import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    import configuration as cfg

    model = get_model_instance_segmentation(num_classes=len(cfg.LABELS) + 1)  # +1 bc of background
    model.to(cfg.DEVICE)
    print(model.roi_heads.box_predictor)