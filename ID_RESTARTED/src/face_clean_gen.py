import os
import cv2
import glob
import shutil
import numpy as np
from tqdm import tqdm

# ================================
# Configurations
# ================================
FINAL_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\faces_clean"   # merged output folder
ID_FACE_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE"     # your folder with ID card face crops
LIVE_FACE_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\LIVE FACE" # your folder with live captured faces
CELEBA_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\img_align_celeba"  # change to your local CelebA folder
MAX_CELEBA_IMAGES = 5000    # limit for quick testing; set None for all

os.makedirs(FINAL_DIR, exist_ok=True)

# ================================
# Helper: Upscale small images
# ================================
def upscale_if_small(img, min_size=150):
    """Upscale image if smaller than min_size in either dimension."""
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img

# ================================
# Step 1: Copy CelebA images
# ================================
def copy_celebA_images():
    celeba_images = sorted(glob.glob(os.path.join(CELEBA_DIR, "*.jpg")))
    if MAX_CELEBA_IMAGES:
        celeba_images = celeba_images[:MAX_CELEBA_IMAGES]
    print(f"Copying {len(celeba_images)} CelebA images...")
    for i, img_path in enumerate(tqdm(celeba_images, desc="CelebA")):
        shutil.copy(img_path, os.path.join(FINAL_DIR, f"celeba_{i:06d}.jpg"))

# ================================
# Step 2: Add LIVE_FACE images
# ================================
def add_live_faces():
    live_images = glob.glob(os.path.join(LIVE_FACE_DIR, "*"))
    print(f"Processing {len(live_images)} LIVE_FACE images...")
    for i, img_path in enumerate(live_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = upscale_if_small(img)
        cv2.imwrite(os.path.join(FINAL_DIR, f"live_{i:03d}.jpg"), img)

# ================================
# Step 3: Add ID_FACE images (extra upscaling)
# ================================
def add_id_faces():
    id_images = glob.glob(os.path.join(ID_FACE_DIR, "*"))
    print(f"Processing {len(id_images)} ID_FACE images...")
    for i, img_path in enumerate(id_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = upscale_if_small(img, min_size=180)  # higher min size for ID crops
        cv2.imwrite(os.path.join(FINAL_DIR, f"id_{i:03d}.jpg"), img)

# ================================
# Main execution
# ================================
if __name__ == "__main__":
    # 1. Copy CelebA images
    copy_celebA_images()

    # 2. Add LIVE_FACE images
    add_live_faces()

    # 3. Add ID_FACE images
    add_id_faces()

    # Summary
    total_images = len(glob.glob(os.path.join(FINAL_DIR, "*.jpg")))
    print(f"\n✅ Merged dataset ready in: {FINAL_DIR}")
    print(f"Total images: {total_images}")