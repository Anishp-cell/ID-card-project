# face_encoder.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import glob

# ==== ResNet-based Encoder ====
class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Updated weights usage
        base_model.fc = nn.Identity()  # remove classifier
        self.backbone = base_model
        self.fc = nn.Linear(512, 128)  # projection to 128-D embedding

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# ==== Preprocessing ====
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ==== Embedding Extraction ====
def extract_embedding(model, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(tensor)
            return embedding.squeeze(0).numpy()
    except Exception as e:
        print(f"[ERROR] Could not process image {img_path}: {e}")
        return None

# ==== Average Embedding for a Folder ====
def get_average_embedding(model, folder_path):
    embeddings = []
    for img_path in glob.glob(f"{folder_path}/*.jpg"):
        emb = extract_embedding(model, img_path)
        if emb is not None and np.all(np.isfinite(emb)):
            embeddings.append(emb)
        else:
            print(f"[WARN] Invalid or empty embedding for: {img_path}")

    if len(embeddings) == 0:
        print(f"[❌ ERROR] No valid embeddings found in folder: {folder_path}")
        return None

    return np.mean(embeddings, axis=0)

# ==== Cosine Similarity ====
def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return None
    return np.dot(a, b) / (norm_a * norm_b)
