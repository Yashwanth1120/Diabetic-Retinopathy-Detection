"""
gradcam.py
----------
A minimal, dependency-light Grad-CAM implementation (Selvaraju et al., 2017).

Grad-CAM works by:
1. Running a forward pass and capturing the activations of the last
   convolutional layer.
2. Running a backward pass from the predicted class score, capturing the
   gradients flowing into that same layer.
3. Using the (global-averaged) gradients as weights for the activations,
   then applying a ReLU, to produce a heatmap that highlights which regions
   of the image mattered most for the prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (1, 3, H, W), already normalized
        class_idx: which class to explain; if None, uses the predicted class
        Returns: heatmap as a (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> one weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))

        return cam, class_idx


def overlay_heatmap(original_image_np, cam, alpha=0.4):
    """
    original_image_np: HxWx3 uint8 RGB image
    cam: HxW float array in [0, 1]
    Returns: HxWx3 uint8 RGB image with heatmap overlay
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + original_image_np * (1 - alpha)).astype(np.uint8)
    return overlay
