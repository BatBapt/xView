import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def get_model_instance_segmentation(num_classes):
    # TODO: rename function
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def faster_rcnn_anchors(num_classes):
    new_anchors_size = ((8,), (16,), (32,), (64,), (128,))  # default from pytorch is ((32,), (64,), (128,), (256,), (512,))

    aspect_ratios = ((0.5, 1.0, 2.0),) * len(new_anchors_size)

    custom_anchor_generator = AnchorGenerator(
        sizes=new_anchors_size,
        aspect_ratios=aspect_ratios
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        rpn_anchor_generator=custom_anchor_generator
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    import configuration as cfg

    model = get_model_instance_segmentation(num_classes=len(cfg.LABELS) + 1)  # +1 bc of background
    model.to(cfg.DEVICE)
    print(model.roi_heads.box_predictor)