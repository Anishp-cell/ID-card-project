import cv2
import os
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import MTCNN

ID_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE"
LIVE_DIR = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\LIVE FACE"
os.makedirs(ID_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)

# --- models ---
yolo = YOLO(r'D:\python\ID CARD DETECTION\ID_RESTARTED\runs\detect\weights\best.pt')
mtcnn = MTCNN(keep_all=False)

# --- camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ---------- helpers ----------
def calculate_sharpness(bgr):
    src = cv2.GaussianBlur(bgr, (5, 5), 0)
    g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def estimate_brightness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    return float(v.mean())

def draw_guide(frame, mode="id"):
    h, w = frame.shape[:2]
    if mode == "id":
        # wide rectangle for ID card (about 1.6:1)
        gw = int(w * 0.55)
        gh = int(gw / 1.6)
    else:
        # square for face
        gh = int(min(h, w) * 0.45)
        gw = gh
    x1 = (w - gw) // 2
    y1 = (h - gh) // 2
    x2 = x1 + gw
    y2 = y1 + gh
    # translucent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    alpha = 0.15
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return (x1, y1, x2, y2)

def box_iou(a, b):
    # a,b = (x1,y1,x2,y2)
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(1, (b[2]-b[0]) * (b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0, inter / area_a if area_a > 0 else 0.0

def put_hud(frame, lines, color=(255,255,255)):
    x, y = 20, 30
    for t in lines:
        cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28

# ---------- ID CAPTURE (guided, press 'q' to accept best) ----------
def capture_id_face():
    print("ID MODE: align the ID card fully within the green box. Press 'q' to save the best frame. Press 'ESC' to cancel.")
    best_id = None
    best_sharp = -1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera read failed.")
            break

        guide = draw_guide(frame, mode="id")
        gx1, gy1, gx2, gy2 = guide

        # detect ID card with YOLO
        result = yolo.predict(source=frame, save=False, conf=0.5, verbose=False)
        boxes = result[0].boxes.xyxy.cpu().numpy() if len(result) > 0 else []

        inside_ok = False
        bright_txt = "N/A"
        sharp_txt  = "N/A"

        # pick the best detection (largest area)
        sel = None
        if len(boxes) > 0:
            areas = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), (b[2]-b[0])*(b[3]-b[1])) for b in boxes]
            sel = max(areas, key=lambda t: t[4])
            x1, y1, x2, y2, _ = sel

            # expand 15% padding
            w = x2 - x1; h = y2 - y1
            pad_x = int(w * 0.15); pad_y = int(h * 0.15)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x); y2 = min(frame.shape[0], y2 + pad_y)

            det = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,180,255), 2)

            iou, cover = box_iou(det, guide)
            inside_ok = cover >= 0.85  # detection mostly inside guide

            # compute metrics on the detected crop
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                bright = estimate_brightness(crop)
                sharp  = calculate_sharpness(crop)
                bright_txt = f"{bright:.0f}"
                sharp_txt  = f"{sharp:.0f}"

                # keep best by sharpness if it is inside the guide and reasonable brightness
                if inside_ok and 70 <= bright <= 200 and sharp > best_sharp:
                    best_sharp = sharp
                    best_id = crop.copy()

        status = []
        status.append("ID CAPTURE")
        status.append("Align ID within the green box. Press 'q' to capture. ESC to cancel.")
        status.append(f"Detection in box: {'YES' if inside_ok else 'NO'}")
        status.append(f"Brightness (V): {bright_txt} (aim ~100–180)")
        status.append(f"Sharpness: {sharp_txt}   Best: {best_sharp:.0f}" if best_sharp > 0 else f"Sharpness: {sharp_txt}")

        put_hud(frame, status, (255,255,255))
        cv2.imshow("ID face capture", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            print("Cancelled.")
            return None
        if k == ord('q'):
            if best_id is not None:
                cv2.imwrite("best_id_card.jpg", best_id)
                print("Best ID card frame saved as best_id_card.jpg")
                return best_id
            else:
                print("No valid ID inside guide yet. Keep the card inside the box and try again.")

# ---------- process ID to 12 face crops (kept from your flow) ----------
def process_id_face():
    img = cv2.imread("best_id_card.jpg")
    if img is None:
        print("[ERROR] ID card image not found.")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is None or len(boxes) == 0:
        print("[ERROR] No face found in ID.")
        return

    x1, y1, x2, y2 = [max(0, int(c)) for c in boxes[0]]
    face_crop = img[y1:y2, x1:x2]

    for i in range(12):
        aug = face_crop.copy()
        brightness = np.random.uniform(0.85, 1.15)
        aug = np.clip(aug * brightness, 0, 255).astype(np.uint8)
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)
        cv2.imwrite(os.path.join(ID_DIR, f"face_{i}.jpg"), aug)

    print("ID face crops saved to", os.path.abspath(ID_DIR))

# ---------- LIVE CAPTURE (guided, press 'q' to save one, ESC to finish) ----------
def capture_live_faces():
    print("LIVE MODE: keep your face in the green box. Press 'q' to save one image (up to 12). Press 'ESC' to finish.")
    count = 0
    while count < 12:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera read failed.")
            break

        guide = draw_guide(frame, mode="face")
        gx1, gy1, gx2, gy2 = guide

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)
        inside_ok = False
        bright_txt = "N/A"
        sharp_txt  = "N/A"

        if boxes is not None:
            # choose largest face
            areas = []
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                areas.append((x1, y1, x2, y2, max(1, (x2-x1)*(y2-y1))))
            x1, y1, x2, y2, _ = max(areas, key=lambda t: t[4])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,150,0), 2)

            det = (x1, y1, x2, y2)
            iou, cover = box_iou(det, guide)
            inside_ok = cover >= 0.85

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                bright = estimate_brightness(crop)
                sharp  = calculate_sharpness(crop)
                bright_txt = f"{bright:.0f}"
                sharp_txt  = f"{sharp:.0f}"

        status = [
            "LIVE CAPTURE",
            "Keep face in box. Press 'q' to save one. ESC to finish.",
            f"Detection in box: {'YES' if inside_ok else 'NO'}",
            f"Brightness (V): {bright_txt} (aim ~100–180)",
            f"Sharpness: {sharp_txt}",
            f"Saved: {count}/12"
        ]
        put_hud(frame, status, (255,255,255))
        cv2.imshow("Live face capture", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == ord('q'):
            if boxes is not None and inside_ok:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    cv2.imwrite(os.path.join(LIVE_DIR, f"live_face_{count}.jpg"), face_crop)
                    count += 1
                    print(f"Saved live face {count}/12")
            else:
                print("Face not fully inside guide. Align and press 'q' again.")

    print("Live face images saved to", os.path.abspath(LIVE_DIR))

# ---------- run ----------
best_id = capture_id_face()
if best_id is not None:
    process_id_face()
    print("Press 'y' + Enter to start live capture, anything else to quit:")
    try:
        if input().strip().lower() == 'y':
            capture_live_faces()
    except EOFError:
        pass

cap.release()
cv2.destroyAllWindows()
