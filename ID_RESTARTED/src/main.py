# run_pipeline.py
import os, glob, cv2, torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import torch.nn.functional as F

# --- import your models & optimizer ---
from residual_attention import PreCNN         # CBAM PreCNN you trained
from siamese_model import SimeseNet           # your Siamese (class name intentionally 'SimeseNet')
from classroom_optimizer import run_demo_optimization
from milp_optimizer import run_milp_demo
from energy_model import calculate_classroom_energy_savings
from gradcam_visualizer import demo_gradcam_visualization

# ---------- CONFIG ----------
YOLO_WEIGHTS   = r"D:\python\ID CARD DETECTION\ID_RESTARTED\runs\detect\weights\best.pt"
CKPT_PRECNN    = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\precnn_runs\precnn.pth"
CKPT_SIAMESE   = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\siamese\siamese.pth"

ID_DIR         = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE"
LIVE_DIR       = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\LIVE FACE"
ID_ENH_DIR     = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\ID FACE ENHACNED"   # keep spelling to match your previous scripts
LIVE_ENH_DIR   = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\LIVE FACE ENHANCED"
COMPARE_DIR    = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\COMPARISON_RESULTS"
IMG_SIZE       = 224
THRESHOLD      = 0.75                 # raise to ~0.75–0.80 if you see false accepts

os.makedirs(ID_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)
os.makedirs(ID_ENH_DIR, exist_ok=True)
os.makedirs(LIVE_ENH_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

# ---------- TRANSFORMS ----------
precnn_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1] range expected by PreCNN
])
siamese_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------- UTILS ----------
def variance_of_laplacian(bgr):
    g = cv2.GaussianBlur(bgr, (5,5), 0)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def estimate_brightness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[...,2].mean())

def draw_guide(frame, mode="id"):
    h, w = frame.shape[:2]
    if mode == "id":
        gw = int(w * 0.55); gh = int(gw / 1.6)  # wide box for ID
    else:
        gh = int(min(h,w) * 0.45); gw = gh      # square for face
    x1 = (w - gw)//2; y1 = (h - gh)//2
    x2 = x1 + gw;    y2 = y1 + gh
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    return (x1,y1,x2,y2)

def iou_cover(det, guide):
    x1,y1,x2,y2 = det
    gx1,gy1,gx2,gy2 = guide
    ix1 = max(x1, gx1); iy1 = max(y1, gy1)
    ix2 = min(x2, gx2); iy2 = min(y2, gy2)
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    inter = iw*ih
    area_det = max(1, (x2-x1)*(y2-y1))
    area_guide = max(1, (gx2-gx1)*(gy2-gy1))
    union = area_det + area_guide - inter
    cover = inter / area_det if area_det>0 else 0.0
    iou = inter / union if union>0 else 0.0
    return iou, cover

def list_images(folder_or_file):
    if os.path.isdir(folder_or_file):
        paths = []
        for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            paths += glob.glob(os.path.join(folder_or_file, e))
        return sorted(paths)
    return [folder_or_file]

def normalize_illum(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(2.0, (8,8)).apply(y)
    return cv2.cvtColor(cv2.merge([y,cr,cb]), cv2.COLOR_YCrCb2BGR)

def center_square_crop(bgr):
    h, w = bgr.shape[:2]
    s = min(h,w)
    y0 = (h - s)//2; x0 = (w - s)//2
    return bgr[y0:y0+s, x0:x0+s]

def enhance_with_precnn(precnn, bgr, device, alpha=0.35):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = precnn_tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = precnn(t)[0]                    # [-1,1], CHW
    out01 = (out*0.5 + 0.5).clamp(0,1)
    out_u8 = (out01*255.0).byte().permute(1,2,0).cpu().numpy()
    enh_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
    
    h, w = enh_bgr.shape[:2]
    orig_bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # Blend 35% enhancement + 65% sharp original crop to preserve details & prevent blur
    blended = cv2.addWeighted(enh_bgr, alpha, orig_bgr, 1.0 - alpha, 0)
    return blended

def embed_once(encoder, img_rgb_u8, device):
    pil = Image.fromarray(img_rgb_u8)
    t = siamese_tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(t)
        z = F.normalize(z, p=2, dim=1)
    return z.squeeze(0)

def embed_tta(encoder, img_rgb_u8, device):
    z1 = embed_once(encoder, img_rgb_u8, device)
    z2 = embed_once(encoder, cv2.flip(img_rgb_u8, 1), device)
    z  = (z1 + z2) / 2
    return F.normalize(z, p=2, dim=0)

def cosine(a, b):
    return float((a*b).sum().cpu())

# ---------- CAPTURE PIPE ----------
def capture_id_and_live():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    yolo = YOLO(YOLO_WEIGHTS)
    mtcnn = MTCNN(keep_all=False)

    # --- ID capture (press q to lock best) ---
    print("[ID] Place ID inside the green box. Press 'q' to accept best frame. 'Esc' to cancel.")
    best_crop = None; best_sharp = -1
    while True:
        ok, frame = cap.read()
        if not ok: print("[ID] Camera read failed"); break
        guide = draw_guide(frame, "id")
        res   = yolo.predict(source=frame, save=False, conf=0.5, verbose=False)
        boxes = res[0].boxes.xyxy.cpu().numpy() if len(res)>0 else []

        inside_ok = False; bright_txt="N/A"; sharp_txt="N/A"
        if len(boxes):
            # choose largest
            x1,y1,x2,y2 = map(int, boxes[np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))][:4])
            # pad 15%
            w,h = (x2-x1),(y2-y1)
            px,py = int(0.15*w), int(0.15*h)
            x1=max(0,x1-px); y1=max(0,y1-py)
            x2=min(frame.shape[1],x2+px); y2=min(frame.shape[0],y2+py)
            det=(x1,y1,x2,y2)
            iou,cover=iou_cover(det, guide)
            inside_ok = cover>=0.85
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,180,255),2)
            crop = frame[y1:y2, x1:x2]
            if crop.size>0:
                bright = estimate_brightness(crop); sharp = variance_of_laplacian(crop)
                bright_txt=f"{bright:.0f}"; sharp_txt=f"{sharp:.0f}"
                if inside_ok and 70<=bright<=200 and sharp>best_sharp:
                    best_sharp=sharp; best_crop=crop.copy()

        cv2.putText(frame, "ID MODE: align ID in box. Press 'q' to capture, 'Esc' to cancel.",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(frame, f"In-box: {'YES' if inside_ok else 'NO'}  Bright: {bright_txt}  Sharp: {sharp_txt}  BestSharp: {best_sharp:.0f}",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow("Capture - ID", frame)
        k=cv2.waitKey(1)&0xFF
        if k==27: best_crop=None; break
        if k==ord('q'):
            if best_crop is not None:
                cv2.imwrite("best_id_card.jpg", best_crop)
                print("[ID] Saved: best_id_card.jpg")
                break
            else:
                print("[ID] No valid crop yet; keep ID fully inside box.")

    # --- Extract face from ID + save 12 crops ---
    if best_crop is not None:
        print("[INFO] Detecting face on ID card...")
        feedback_img = best_crop.copy()
        rgb = cv2.cvtColor(feedback_img, cv2.COLOR_BGR2RGB)
        
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            x1, y1, x2, y2 = [max(0, int(c)) for c in boxes[0]]
            cv2.rectangle(feedback_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print("[INFO] Face detected. Showing result for 3 seconds.")
            
            face = best_crop[y1:y2, x1:x2]
            if face.size > 0 and (face.shape[0] < 224 or face.shape[1] < 224):
                face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            for i in range(12):
                aug = face.copy()
                aug = np.clip(aug * np.random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
                if np.random.rand() > 0.5:
                    aug = cv2.flip(aug, 1)
                cv2.imwrite(os.path.join(ID_DIR, f"face_{i}.jpg"), aug)
            print(f"[ID] Saved 12 crops to {os.path.abspath(ID_DIR)}")

        else:
            print("[WARN] No face detected on the ID card.")

        cv2.imshow("ID Face Detection Feedback", feedback_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    # --- LIVE capture (press q to save one; up to 12; Esc to finish) ---
    print("[LIVE] Keep your face in the box. Press 'q' to save (up to 12). 'Esc' to finish.")
    saved=0
    mtcnn_live = MTCNN(keep_all=True)
    while saved<12:
        ok, frame = cap.read()
        if not ok: print("[LIVE] Camera read failed"); break
        guide = draw_guide(frame, "face")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes,_=mtcnn_live.detect(rgb)
        inside_ok=False
        if boxes is not None:
            # largest face
            areas=[(int(b[0]),int(b[1]),int(b[2]),int(b[3]), (b[2]-b[0])*(b[3]-b[1])) for b in boxes]
            x1,y1,x2,y2,_= max(areas, key=lambda t:t[4])
            det=(x1,y1,x2,y2)
            _,cover=iou_cover(det, guide)
            inside_ok=cover>=0.85
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,150,0),2)

        cv2.putText(frame, f"[LIVE] In-box: {'YES' if inside_ok else 'NO'}  Saved: {saved}/12",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow("Capture - LIVE", frame)
        k=cv2.waitKey(1)&0xFF
        if k==27: break
        if k==ord('q') and inside_ok:
            crop=frame[y1:y2, x1:x2]
            if crop.size>0:
                cv2.imwrite(os.path.join(LIVE_DIR, f"live_face_{saved}.jpg"), crop)
                saved+=1
                print(f"[LIVE] Saved {saved}/12")

    cap.release(); cv2.destroyAllWindows()

def enhance_folder(precnn, device, in_dir, out_dir):
    paths = list_images(in_dir)
    if not paths:
        print(f"[ENH] No images in {os.path.abspath(in_dir)}"); return
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None: continue
        bgr = normalize_illum(bgr)
        bgr = center_square_crop(bgr)
        enh = enhance_with_precnn(precnn, bgr, device)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(out_dir, name), enh)
    print(f"[ENH] Wrote enhanced crops to {os.path.abspath(out_dir)}")

def calculate_match_probability(sim: float):
    """
    Calibrated Probabilistic Decision Model:
    Converts cosine similarity into a calibrated probability P(Match) (%)
    using a sigmoid activation centered at s0=0.65 with steepness k=20.
    """
    s0 = 0.65
    k = 20.0
    prob = 1.0 / (1.0 + np.exp(-k * (sim - s0)))
    prob_pct = float(prob * 100.0)

    if sim >= 0.70 or prob_pct >= 73.0:
        verdict = "SAME person"
        confidence_level = "HIGH CONFIDENCE MATCH"
    elif sim >= 0.58 or prob_pct >= 20.0:
        verdict = "PROBABLE MATCH"
        confidence_level = "MODERATE CONFIDENCE (REVIEW SUGGESTED)"
    else:
        verdict = "DIFFERENT people"
        confidence_level = "HIGH CONFIDENCE REJECTION"

    return prob_pct, verdict, confidence_level

def save_comparison_visualization(sim, verdict, prob_pct, confidence_level):
    id_paths = list_images(ID_ENH_DIR)
    live_paths = list_images(LIVE_ENH_DIR)
    
    if not id_paths or not live_paths:
        print("[WARN] Could not find enhanced ID or Live images for visual comparison.")
        return
        
    id_img = cv2.imread(id_paths[0])
    live_img = cv2.imread(live_paths[0])
    
    if id_img is None or live_img is None:
        return
        
    size = (300, 300)
    id_img = cv2.resize(id_img, size)
    live_img = cv2.resize(live_img, size)
    
    canvas_w = 640
    canvas_h = 440
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 30
    
    canvas[90:390, 15:315] = id_img
    canvas[90:390, 325:625] = live_img
    
    if verdict == "SAME person":
        match_color = (0, 255, 0)
    elif verdict == "PROBABLE MATCH":
        match_color = (0, 215, 255) # Yellow/Orange
    else:
        match_color = (0, 0, 255)
    
    cv2.putText(canvas, f"Verdict: {verdict}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, match_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Match Probability: {prob_pct:.1f}% ({confidence_level})", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Cosine Similarity: {sim:.4f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, "ID Card Face (Enhanced)", (55, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Live Face (Enhanced)", (365, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    
    out_path = os.path.join(COMPARE_DIR, "manual_verification_comparison.jpg")
    cv2.imwrite(out_path, canvas)
    
    root_out_path = r"D:\python\ID CARD DETECTION\last_verification_result.jpg"
    cv2.imwrite(root_out_path, canvas)
    
    print(f"[VERIFY] Saved side-by-side comparison image to:")
    print(f"  -> {os.path.abspath(out_path)}")
    print(f"  -> {os.path.abspath(root_out_path)}")
    
    cv2.imshow("Identity Verification Side-by-Side Check", canvas)
    cv2.waitKey(4000)
    cv2.destroyAllWindows()

def mean_embedding_from_path(encoder, precnn, device, p, enhance_both=True):
    paths = list_images(p)
    zs=[]
    for fp in paths:
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None: continue
        bgr = normalize_illum(bgr)
        bgr = center_square_crop(bgr)
        if enhance_both:
            enh_rgb = cv2.cvtColor(enhance_with_precnn(precnn, bgr, device), cv2.COLOR_BGR2RGB)
        else:
            enh_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        zs.append(embed_tta(encoder, enh_rgb, device))
    return torch.stack(zs, dim=0).mean(dim=0)

def run_siamese_compare(device):
    # load models
    precnn = PreCNN(in_ch=3, mid_ch=48, num_blocks=6).to(device).eval()
    precnn.load_state_dict(torch.load(CKPT_PRECNN, map_location=device))

    snet = SimeseNet(embed_dim=128).to(device)
    snet.load_state_dict(torch.load(CKPT_SIAMESE, map_location=device))
    snet.eval()
    encoder = snet.encoder
    encoder.eval()

    # compute embeddings (enhance in-memory for BOTH sides; disk save is for both)
    z_id   = mean_embedding_from_path(encoder, precnn, device, ID_DIR,   enhance_both=True)
    z_live = mean_embedding_from_path(encoder, precnn, device, LIVE_DIR, enhance_both=True)

    sim = cosine(z_id, z_live)
    prob_pct, verdict, confidence_level = calculate_match_probability(sim)
    
    print(f"\n[DECISION ALGO] Cosine Similarity: {sim:.4f}")
    print(f"[DECISION ALGO] Calibrated Match Probability: {prob_pct:.2f}%")
    print(f"[DECISION ALGO] Confidence Classification: {confidence_level}")
    print(f"[DECISION ALGO] Final Verdict: ==> {verdict} <==")
    
    save_comparison_visualization(sim, verdict, prob_pct, confidence_level)
    return sim, verdict

# ---------- MAIN ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 1) Capture ID + LIVE
    capture_id_and_live()

    # 2) Enhance ID and LIVE crops to disk for manual inspection
    precnn_for_save = PreCNN(in_ch=3, mid_ch=48, num_blocks=6).to(device).eval()
    precnn_for_save.load_state_dict(torch.load(CKPT_PRECNN, map_location=device))
    enhance_folder(precnn_for_save, device, ID_DIR, ID_ENH_DIR)
    enhance_folder(precnn_for_save, device, LIVE_DIR, LIVE_ENH_DIR)

    # 3) Siamese comparison & save visual side-by-side comparison collage
    sim, verdict = run_siamese_compare(device)

    # 4) Dynamic Classroom Scheduling & Lecture Merging Optimization
    print("\n[OPTIMIZER] Executing Dynamic Classroom Allocation & Lecture Merging Engine...")
    run_demo_optimization(live_attendance_count=15)

    # 5) MILP Mathematical Constraint Optimization & ASHRAE Energy Savings Evaluation
    print("\n[MATH OPTIMIZER] Running Formal MILP Solver (Equations 1-5)...")
    milp_res = run_milp_demo()
    vacated_count = len(milp_res.get("vacated_rooms", []))
    
    print("\n[ENERGY EVALUATION] Calculating ASHRAE Thermodynamic Energy & Cost Savings...")
    e_savings = calculate_classroom_energy_savings(vacated_rooms_count=max(1, vacated_count), duration_hours=1.0)
    print(f" -> Thermal Load Saved    : {e_savings['total_thermal_load_saved_kw']} kW")
    print(f" -> Electrical Energy     : {e_savings['electrical_energy_saved_kwh']} kWh saved per lecture hour")
    print(f" -> Estimated Cost Saved  : ${e_savings['cost_savings_usd']} USD (₹{e_savings['cost_savings_inr']} INR)")

    # 6) Model Explainability & Feature Heatmap Generation
    print("\n[EXPLAINABILITY] Generating Grad-CAM Feature Heatmaps for PreCNN-CBAM Block...")
    demo_gradcam_visualization()
