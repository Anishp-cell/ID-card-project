import torch 
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_dim=128):   
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(512, embed_dim)
    
    def forward(self, x):
        features = self.backbone(x)
        z = self.proj(features)
        z = f.normalize(input=z, p=2, dim=1)
        return z

class SimeseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
    def forward(self,x1,x2):
        embedding1= self.encoder(x1)
        embedding2= self.encoder(x2)
        return embedding1, embedding2
def contrastive_loss(embedding1, embedding2, label,margin=1.0):
        eucladian_distance= f.pairwise_distance(embedding1, embedding2, keepdim=False)
        positive_mathc= label*(eucladian_distance**2)
        negative_nomatch= (1-label)*torch.clamp(margin-eucladian_distance, min=0)**2
        return (positive_mathc+ negative_nomatch).mean()
def cosine_sim(embedding1,embeddig2):
        return f.cosine_similarity(embedding1, embeddig2, dim=1)
    