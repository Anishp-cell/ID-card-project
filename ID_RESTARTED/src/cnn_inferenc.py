# enhance_with_precnn.py
import os
import glob
import cv2
import torch
from torchvision import transforms
from residual_attention import PreCNN  # corrected import
from PIL import Image

CKPT = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\precnn_runs\precnn.pth"
IN_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE"
OUT_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE ENHACNED"

os.makedirs(OUT_DIR, exist_ok=True)

to_t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def tensor_to_bgr_uint8(x):
    x = (x * 0.5 + 0.5).clamp(0,1)
    x = (x*255.0).permute(1,2,0).cpu().numpy().astype("uint8")
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PreCNN(in_ch=3, mid_ch=48, num_blocks=6).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")) + glob.glob(os.path.join(IN_DIR, "*.png")))
    if not paths:
        print(f"No images in {IN_DIR}. Put ID/live crops here.")
        return

    with torch.no_grad():
        for p in paths:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            t = to_t(pil)

            # forward
            pred = model(t.unsqueeze(0).to(device))[0]
            out = tensor_to_bgr_uint8(pred)

            name = os.path.basename(p)
            cv2.imwrite(os.path.join(OUT_DIR, name), out)
            print(f"saved: {os.path.join(OUT_DIR, name)}")

if __name__ == "__main__":
    main()
