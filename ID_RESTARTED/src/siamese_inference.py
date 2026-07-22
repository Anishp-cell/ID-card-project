import torch
import torch.nn.functional as F
import cv2, os, glob
from PIL import Image
from torchvision import transforms

from residual_attention import PreCNN
from siamese_model import SimeseNet

CKPT_PRECNN  = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\precnn_runs\precnn.pth"
CKPT_SIAMESE = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\siamese\siamese.pth"
IMG_SIZE = 224
THRESHOLD = 0.80

precnn_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
siamese_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def enhance_with_precnn(model, bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = precnn_tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)[0]
    out01 = (out * 0.5 + 0.5).clamp(0,1)
    out_u8 = (out01 * 255.0).byte().permute(1,2,0).cpu().numpy()
    return out_u8

def embed(encoder, img_rgb_u8, device):
    pil = Image.fromarray(img_rgb_u8)
    t = siamese_tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(t)
        z = F.normalize(z, p=2, dim=1)
    return z.squeeze(0)

def cosine(a, b):
    return float((a * b).sum().cpu())

def load_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(os.path.abspath(path))
    return bgr

def list_images(p):
    if os.path.isdir(p):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(p, e))
        if not paths:
            raise FileNotFoundError(f"No images found in folder: {os.path.abspath(p)}")
        return paths
    else:
        return [p]

def mean_embedding_from_path(encoder, p, precnn, device):
    paths = list_images(p)
    zs = []
    for fp in paths:
        bgr = load_image(fp)
        enh = enhance_with_precnn(precnn, bgr, device)
        zs.append(embed(encoder, enh, device))
    return torch.stack(zs, dim=0).mean(dim=0)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    precnn = PreCNN(in_ch=3, mid_ch=48, num_blocks=6).to(device).eval()
    precnn.load_state_dict(torch.load(CKPT_PRECNN, map_location=device))

    snet = SimeseNet(embed_dim=128).to(device)
    snet.load_state_dict(torch.load(CKPT_SIAMESE, map_location=device))
    snet.eval()
    encoder = snet.encoder
    encoder.eval()

    ID_PATH   = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE ENHACNED"
    LIVE_PATH = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\LIVE FACE"

    print("ID source:", os.path.abspath(ID_PATH))
    print("LIVE source:", os.path.abspath(LIVE_PATH))

    z_id_mean   = mean_embedding_from_path(encoder, ID_PATH,   precnn, device)
    z_live_mean = mean_embedding_from_path(encoder, LIVE_PATH, precnn, device)

    sim = cosine(z_id_mean, z_live_mean)
    print(f"Cosine similarity: {sim:.4f} (thr={THRESHOLD:.2f})")
    print("Verified:", "SAME person" if sim >= THRESHOLD else "DIFFERENT people")
