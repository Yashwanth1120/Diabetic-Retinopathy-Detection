"""
model.py
--------
Transfer-learning model definitions. We reuse a backbone pretrained on
ImageNet and replace its final classification layer with one matching our
number of DR severity classes.
"""

import torch.nn as nn
from torchvision import models


def build_model(backbone="efficientnet_b0", num_classes=5, freeze_backbone=False):
    """
    backbone: "efficientnet_b0" or "resnet50"
    freeze_backbone: if True, only the final layer is trained (faster, but
        usually lower accuracy). Fine-tuning the whole network (False) is
        recommended once your pipeline works end-to-end.
    """
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Use 'efficientnet_b0' or 'resnet50'.")

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Keep the final layer trainable; freeze everything else
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    return model


def get_target_layer(model, backbone):
    """
    Returns the last convolutional layer, used by Grad-CAM to compute
    activation maps. Different architectures name this differently.
    """
    if backbone == "efficientnet_b0":
        return model.features[-1]
    elif backbone == "resnet50":
        return model.layer4[-1]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
