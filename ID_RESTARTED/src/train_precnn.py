from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
import glob
import cv2
import random
import os
import pandas as pd
from residual_attention import PreCNN
from siamese_model import SimeseNet

data_dir = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\faces_clean"
save_dir = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\precnn_runs"
CKPT_SIAMESE = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\siamese\siamese.pth"
CKPT = os.path.join(save_dir, "precnn.pth")
sample_dir = os.path.join(save_dir, "samples")
log_file = os.path.join(save_dir, "precnn_training_log.csv")

batchsize = 16
epochs = 20
lr = 1e-3
img_size = 224
num_workers = 0  # Set to 0 for Windows compatibility
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)


def to_tensor_normalization(img_pil):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tfm(img_pil)


def denormalize_to_uint8(x):
    x = (x*0.5 + 0.5).clamp(0., 1)
    x = (x*255.0).round().byte()
    return x


def save_sample_grid(pred, clean, degr, path):
    with torch.no_grad():
        b = min(4, pred.size(0))
        pred_u8 = denormalize_to_uint8(pred[:b].cpu())
        clean_u8 = denormalize_to_uint8(clean[:b].cpu())
        degr_u8 = denormalize_to_uint8(degr[:b].cpu())

        rows = []
        for i in range(b):
            trip = torch.cat([degr_u8[i], pred_u8[i], clean_u8[i]], dim=2)
            rows.append(trip)
        grid = torch.cat(rows, dim=1)
        grid = grid.permute(1, 2, 0).numpy()
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, grid_bgr)


def jpeg_compress(img_bgr, quality):
    encode_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encode = cv2.imencode('.jpg', img_bgr, encode_parameters)
    decode = cv2.imdecode(encode, cv2.IMREAD_COLOR)
    return decode


def degrade_bgr(img_bgr):
    height, width = img_bgr.shape[:2]
    if random.random() < 0.7:
        a = random.choice([3, 5, 7])
        img_bgr = cv2.GaussianBlur(img_bgr, (a, a), 0)
    if random.random() < 0.7:
        noise = np.random.randn(height, width, 3) * random.uniform(5, 15)
        img_bgr = np.clip(img_bgr.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    if random.random() < 0.7:
        q = random.randint(20, 60)
        img_bgr = jpeg_compress(img_bgr=img_bgr, quality=q)
    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-8, 8)
        img_bgr = np.clip(alpha * img_bgr+beta, 0, 255).astype(np.uint8)

    return img_bgr


class FaceRestorationDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            glob.glob(os.path.join(root, "*.jpg")) +
            glob.glob(os.path.join(root, "*.jpeg")) +
            glob.glob(os.path.join(root, "*.png"))
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pah = self.paths[idx]
        clean_bgr = cv2.imread(pah, cv2.IMREAD_COLOR)
        if clean_bgr is None:
            clean_bgr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        degraded_bgr = degrade_bgr(clean_bgr)
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
        clean_pil = Image.fromarray(clean_rgb)
        degrade_pil = Image.fromarray(degraded_rgb)
        clean_tens = to_tensor_normalization(clean_pil)
        degrade_tens = to_tensor_normalization(degrade_pil)
        return degrade_tens, clean_tens


class VGGPerceptualloss(nn.Module):
    def __init__(self):
        super().__init__()
        vggmodel = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vggmodel.parameters():
            p.requires_grad = False
        self.slice = nn.Sequential(*[vggmodel[i] for i in range(17)])
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x*0.5+0.5).clamp(0, 1)
        y = (y*0.5+0.5).clamp(0, 1)
        x = (x-self.mean) / self.std
        y = (y-self.mean) / self.std
        fx = self.slice(x)
        fy = self.slice(y)
        return f.l1_loss(fx, fy)


def edge_map(t):
    t01 = (t*0.5 + 0.5).clamp(0, 1)
    r, g, b = t01[:, 0:1], t01[:, 1:2], t01[:, 2:3]
    gray = 0.299*r + 0.587*g + 0.114*b
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                      dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                      dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
    gx = f.conv2d(gray, kx, padding=1)
    gy = f.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return mag


def edge_loss(pred, target):
    ep = edge_map(pred)
    et = edge_map(target)
    return f.l1_loss(ep, et)


def main():
    print(f"Starting training on device: {device}")
    ds = FaceRestorationDataset(data_dir)
    if len(ds) == 0:
        print(
            f"No images found in {data_dir}. Put cropped images of faces in this directory")
        return
    print(f"Found {len(ds)} images for training.")

    # Load frozen Siamese encoder for identity loss
    snet = SimeseNet(embed_dim=128).to(device).eval()
    snet.load_state_dict(torch.load(CKPT_SIAMESE, map_location=device))
    encoder = snet.encoder
    for param in encoder.parameters():
        param.requires_grad = False
    print("Loaded frozen Siamese encoder for identity loss.")

    dataloader = DataLoader(
        dataset=ds,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True
    )
    model = PreCNN(in_ch=3, mid_ch=48, num_blocks=6).to(device)
    optimize = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    perc = VGGPerceptualloss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_loss = float("inf")

    log_df = pd.DataFrame(
        columns=['epoch', 'total_loss', 'l1_loss', 'perceptual_loss', 'edge_loss', 'identity_loss'])

    for epoch in range(1, epochs+1):
        model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'p': 0, 'e': 0, 'id': 0}

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for i, (deg, clean) in enumerate(progress_bar):
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(deg)

                # Reconstruction losses
                loss_l1 = l1_loss(pred, clean)
                loss_p = perc(pred, clean)
                loss_e = edge_loss(pred, clean)

                # Identity loss
                with torch.no_grad():
                    z_clean = encoder(clean)
                z_pred = encoder(pred)
                loss_id = f.l1_loss(z_pred, z_clean)

                # Total loss
                loss = loss_l1 + 0.1*loss_p + 0.05*loss_e + 0.2*loss_id

            optimize.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimize)
            scaler.update()

            epoch_losses['total'] += loss.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['p'] += loss_p.item()
            epoch_losses['e'] += loss_e.item()
            epoch_losses['id'] += loss_id.item()

            progress_bar.set_postfix(
                loss=epoch_losses['total']/(i+1), id_loss=epoch_losses['id']/(i+1))

        avg_total_loss = epoch_losses['total'] / len(dataloader)
        avg_l1_loss = epoch_losses['l1'] / len(dataloader)
        avg_p_loss = epoch_losses['p'] / len(dataloader)
        avg_e_loss = epoch_losses['e'] / len(dataloader)
        avg_id_loss = epoch_losses['id'] / len(dataloader)

        print(
            f"Epoch {epoch} summary: Avg Loss: {avg_total_loss:.4f}, Identity Loss: {avg_id_loss:.4f}")

        # Log to DataFrame
        log_df.loc[epoch-1] = [epoch, avg_total_loss, avg_l1_loss, avg_p_loss, avg_e_loss, avg_id_loss]
        log_df.to_csv(log_file, index=False)

        model.eval()
        with torch.no_grad():
            sample_deg, sample_clean = next(iter(dataloader))
            sample_deg = sample_deg.to(device)
            sample_pred = model(sample_deg)
            save_sample_grid(
                sample_pred, sample_clean.to(device), sample_deg,
                os.path.join(sample_dir, f"epoch_{epoch:03d}.jpg")
            )

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), CKPT)
            print(f"Saved checkpoint to: {CKPT}")

if __name__ == "__main__":
    main()
