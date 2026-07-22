# Intelligent ID Card Detection, Face Verification & Dynamic Classroom Optimization System

An end-to-end AI-driven smart campus system that combines computer vision for biometric attendance verification with an intelligent operations engine for **dynamic classroom scheduling and lecture merging**.

---

## 📌 Project Overview

Traditional attendance systems focus solely on passive logging. This system bridges biometric computer vision with smart building operations by using verified attendance to dynamically optimize classroom occupancy, reduce energy waste, and save faculty hours.

### Key Highlights
- **Automated ID Card Detection**: Fine-tuned YOLOv11 object detector for robust ID card detection.
- **Attention-Based Face Restoration (PreCNN)**: Uses Convolutional Block Attention Modules (CBAM) to enhance low-resolution, blurred, or compressed face crops extracted from ID photos.
- **Deep Face Verification**: ResNet-18 based Siamese Neural Network trained with Contrastive Loss to compute 128-D facial embeddings and perform cosine similarity matching against live webcam captures.
- **Dynamic Lecture Merging Engine**: Automatically identifies low-attendance parallel lecture sections, verifies classroom capacity constraints, reallocates sessions to a single room, vacates redundant spaces, and generates automated notification/energy-saving payloads.

---

## 🏗️ System Architecture

```text
               +----------------------------------+
               |     Camera Input (ID / Live)     |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   ID Detection (YOLOv11 Model)   |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  Face Crop Extraction (MTCNN)    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Face Enhancement (CBAM PreCNN)   |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Siamese Embedding Matching (z)   |
               +----------------------------------+
                                |
                                v
         +----------------------------------------------+
         | Verified Attendance & Occupancy Calculation  |
         +----------------------------------------------+
                                |
                                v
         +----------------------------------------------+
         | Dynamic Lecture Merging Optimization Engine  |
         +----------------------------------------------+
                                |
       +------------------------+------------------------+
       |                        |                        |
       v                        v                        v
+------------------+   +-------------------+   +--------------------+
|  Faculty Alert   |   |   Student Alert   |   | Facilities/Admin   |
| (Room Shifted)   |   |  (New Classroom)  |   | (Power-off Vacated)|
+------------------+   +-------------------+   +--------------------+
```

---

## 📁 Repository Structure

```text
.
├── README.md                          # Comprehensive project documentation
├── details of project.txt             # Original specifications & project notes
├── .gitignore                         # Configured git exclusions
└── ID_RESTARTED/
    ├── config/
    │   └── config.yaml                # YOLOv11 dataset configuration
    ├── models/
    │   └── haarcascade_frontalface_default.xml # OpenCV Cascade classifier fallback
    ├── runs/                          # Trained model checkpoints (ignored by Git)
    │   └── detect/weights/best.pt     # Fine-tuned YOLOv11 ID card weights
    └── src/
        ├── main.py                    # Main pipeline orchestrator & demo
        ├── classroom_optimizer.py     # Dynamic Lecture Merging & Scheduling Engine
        ├── residual_attention.py      # PreCNN Attention Model (CBAM + Residual blocks)
        ├── siamese_model.py           # Siamese Neural Network (Encoder + Contrastive Loss)
        ├── siamese_inference.py       # Offline face verification benchmark script
        ├── cnn_inferenc.py            # PreCNN enhancement inference utility
        ├── dataset_pairer.py          # PyTorch dataset pair generator for SiameseNet
        ├── face_clean_gen.py          # Dataset aggregator for PreCNN training
        ├── face_encoder.py            # Standalone ResNet-18 feature extractor
        ├── label&split.py             # YOLO annotation processor & split generator
        ├── save_img_&_label.py        # YOLO folder formatting utility
        ├── train_model.py             # Siamese network training script
        ├── train_precnn.py            # PreCNN restoration training script
        ├── yolov11.py                 # YOLOv11 GPU training script
        └── inference/
            └── capture.py             # Webcam guide HUD & image capture script
```

---

## ⚡ Key Innovations & Modules

### 1. PreCNN Attention-Based Image Restoration (`residual_attention.py`)
- Incorporates spatial and channel attention mechanisms (**CBAM**) combined with residual convolutional blocks (`ResidualCBAM`).
- Enhances degraded, small, or compressed face crops from ID card photos to preserve identity-critical features.

### 2. Siamese Verification Network (`siamese_model.py`)
- Employs a ResNet-18 backbone projection head mapping face images into a 128-dimensional L2-normalized embedding space.
- Optimized using **Contrastive Loss** to pull positive facial pairs together and push negative pairs apart.

### 3. Dynamic Lecture Merging Engine (`classroom_optimizer.py`)
- **Attendance Rate Thresholding**: Continuously monitors class sessions; flags sessions falling under an occupancy threshold (e.g., $<35\%$).
- **Compatibility Matching**: Automatically pairs candidates matching the same subject, timeslot, and semester.
- **Capacity Constraint Satisfaction**: Ensures the combined headcount fits into target room capacity before issuing reassignments.
- **Facility Automation Payload**: Calculates estimated HVAC and lighting energy savings ($kWh$) and emits notification events for Faculty, Students, and Facilities Admin.

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** (Optional for acceleration; supports RTX 50-series / Blackwell architecture via PyTorch Nightly cu128)

### 1. Clone the Repository
```bash
git clone https://github.com/Anishp-cell/ID-card-project.git
cd ID-card-project
```

### 2. Set Up Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
# Standard PyTorch installation
pip install torch torchvision ultralytics facenet-pytorch opencv-python pillow pandas scikit-learn tqdm pyyaml

# For NVIDIA RTX 50-series / Blackwell GPUs (CUDA 12.8 support):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

---

## 💻 Usage

### 1. Run Full End-to-End Pipeline
Runs ID card detection, interactive webcam capture, PreCNN restoration, Siamese verification, and the Dynamic Classroom Optimization engine:
```bash
python ID_RESTARTED/src/main.py
```

### 2. Test Dynamic Classroom Optimizer Standalone
Simulates multi-classroom low-attendance scenarios and outputs merge recommendations, notifications, and energy savings:
```bash
python ID_RESTARTED/src/classroom_optimizer.py
```

### 3. Train YOLOv11 ID Card Detector (GPU)
Fine-tunes YOLOv11 on custom ID card dataset:
```bash
python ID_RESTARTED/src/yolov11.py
```

---

## 📊 Sample Output

```text
======================================================================
=== DYNAMIC CLASSROOM LECTURE MERGING SYSTEM RESULT ===
======================================================================
Attendance Threshold: 35.0%

Current Session Status:
 - [SESS_01] Data Structures (Div-A): 15/60 present (25.0%) -> LOW ATTENDANCE (MERGE CANDIDATE)
 - [SESS_02] Data Structures (Div-B): 18/60 present (30.0%) -> LOW ATTENDANCE (MERGE CANDIDATE)
 - [SESS_03] Computer Networks (Div-C): 52/60 present (86.7%) -> NORMAL

Generated Merge Recommendations:

Recommendation #1:
  Merged Sessions : ['SESS_01', 'SESS_02'] (Div-A, Div-B)
  Subject         : Data Structures
  Target Room     : Room A101 (Capacity: 60)
  Combined Count  : 33 students
  Freed Room(s)   : Room A102
  Assigned Faculty: Dr. Sharma
  Est. Energy Save: 1.5 kWh

Notifications Sent:
 - [FACULTY NOTIFICATION] Dr. Sharma: Your Data Structures lecture for Divisions (Div-A, Div-B) at 10:00-11:00 is MERGED into Room Room A101.
 - [STUDENT NOTIFICATION] Attention Divisions Div-A, Div-B: Data Structures lecture at 10:00-11:00 is shifted to Room Room A101 due to dynamic scheduling.
 - [ADMIN / FACILITIES] Room(s) Room A102 VACATED at 10:00-11:00. Estimated energy saved: 1.5 kWh. HVAC & Lighting powering off.
======================================================================
```

---

## 📜 License & Citation

Distributed for academic and research purposes. If using this codebase or architecture for research papers, please reference this repository.
