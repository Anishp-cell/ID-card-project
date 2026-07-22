"""
Grad-CAM Feature Heatmap Generator for CBAM PreCNN Explainability
------------------------------------------------------------------
Computes Gradient-Weighted Class Activation Mapping (Grad-CAM) activations
over the final CBAM residual convolutional block of PreCNN.

Equations:
  alpha_k = (1/Z) * sum_{i,j} (d Y / d A_{i,j}^k)
  M_GradCAM = ReLU( sum_k alpha_k * A^k )
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from residual_attention import PreCNN

class PreCNNGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        # Use sum of high-level feature responses as target output Y
        loss = output.pow(2).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Neuron importance weights alpha_k via global average pooling
        grads = self.gradients.cpu().data.numpy()[0]          # C x H x W
        target_acts = self.activations.cpu().data.numpy()[0]  # C x H x W

        weights = np.mean(grads, axis=(1, 2))                # C
        cam = np.zeros(target_acts.shape[1:], dtype=np.float32) # H x W

        for i, w in enumerate(weights):
            cam += w * target_acts[i, :, :]

        cam = np.maximum(cam, 0)  # ReLU to isolate positive contributions
        if cam.max() > 0:
            cam = cam / cam.max() # Normalize to [0, 1]

        return cam

def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay

def demo_gradcam_visualization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PreCNN(in_ch=3, mid_ch=48, num_blocks=6).to(device)
    
    # Target the last ResidualCBAM block in model.body
    target_layer = model.body[-1]
    grad_cam = PreCNNGradCAM(model, target_layer)

    # Dummy input face image tensor
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_t = tfm(Image.fromarray(dummy_img)).unsqueeze(0).to(device)

    heatmap = grad_cam.generate_heatmap(input_t)
    overlay = overlay_heatmap(dummy_img, heatmap)

    print("\n" + "="*70)
    print("=== GRAD-CAM CBAM MODEL EXPLAINABILITY GENERATOR ===")
    print("="*70)
    print(f"Heatmap Computed Successfully. Resolution: {heatmap.shape}")
    print(f"Heatmap Intensity Range: min={heatmap.min():.4f}, max={heatmap.max():.4f}")
    print(f"Target Layer Tracked: {target_layer.__class__.__name__}")
    print("="*70 + "\n")

    return heatmap, overlay

if __name__ == "__main__":
    demo_gradcam_visualization()
