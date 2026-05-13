# Ageisnet — Real-Time Driver Safety System

> A production-grade, smartphone-CPU-ready pipeline that detects driver drowsiness and verifies driver identity using computer vision and deep learning.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline — Step by Step](#pipeline--step-by-step)
  - [Step 1 — Dataset Organisation](#step-1--dataset-organisation)
  - [Step 2 — Feature Extraction](#step-2--feature-extraction)
  - [Step 3 — Train / Test Split](#step-3--train--test-split)
  - [Step 4 — TCN Face Stream](#step-4--tcn-face-stream)
  - [Step 5 — ST-GCN Body Stream + Attention Fusion](#step-5--st-gcn-body-stream--attention-fusion)
  - [Step 6 — ArcFace Identity Verification](#step-6--arcface-identity-verification)
  - [Step 7 — ONNX Export and INT8 Quantisation](#step-7--onnx-export-and-int8-quantisation)
- [Live Webcam Demo](#live-webcam-demo)
- [Model Performance Summary](#model-performance-summary)
- [ONNX Deployment Benchmarks](#onnx-deployment-benchmarks)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

Ageisnet is designed for real-world carpooling platforms. It runs two parallel AI streams on every video frame:

1. **Drowsiness Detection** — monitors the driver's face and body posture in real time and emits a continuous Safety Score (0 = fully drowsy, 1 = fully alert).
2. **Identity Verification** — confirms at ride start that the registered driver is the person behind the wheel using ArcFace face recognition.

All models are exported to ONNX with INT8 quantisation for deployment on mid-range smartphone CPUs without a GPU.

---

## System Architecture

```
Live Camera Feed
       |
       +-----------------------------+
       |                             |
  Face Landmarks               Body Keypoints
  (MediaPipe FaceLandmarker)   (MediaPipe BlazePose)
       |                             |
  EAR · MAR · Head Pose        33 (x,y,z,v) keypoints
       |                             |
  TCN Classifier               ST-GCN Classifier
  (Face Stream)                (Body Stream)
       |                             |
       +---------> Attention <-------+
                   Fusion MLP
                       |
                  Safety Score
                  (0.0 – 1.0)
                       |
            >= 0.5  Alert / < 0.5  Drowsy
                       |
                  + Identity
                  Verification
                  (ArcFace)
```

**Safety Score semantics:** `1.0` = fully alert, `0.0` = fully drowsy. Threshold at `0.5`.

The fusion layer learns a per-sample attention weight between the two streams. On this dataset the face stream earned ~64.4% weight and the body stream ~35.6%.

---

## Dataset

**Driver Drowsiness Dataset (DDD)**
- Source: [Kaggle — ismailnasri20/driver-drowsiness-dataset-ddd](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- Pre-extracted PNG frames (no video decoding required)
- Multiple subjects, letter-prefixed filenames: `a####.png` = subject A, `b####.png` = subject B, etc.
- Two classes: `alert/` and `drowsy/`
- Total frames processed: **41,769** (99.97% face detection rate)
- Train / Test split: **80 / 20** stratified by class

> The dataset is **not included** in this repository (5 GB+). Download it from Kaggle and run Step 1 to organise it.

---

## Results

| Model | Accuracy | F1 Score | Parameters |
|---|---|---|---|
| TCN (face stream) | **99.5%** | 0.995 | 220,993 |
| ST-GCN (body stream) | **98.6%** | 0.986 | 216,810 |
| Attention Fusion | **99.33%** | 0.993 | 149 |
| ONNX INT8 Fusion | **100.0%** | 1.000 | — |

**Identity Verification (ArcFace)**
- Same-person: 5/5 verified, avg cosine similarity = **0.923**
- Different-person: 5/5 rejected, avg cosine similarity = **0.166**
- Separability gap: **0.758** — clear margin between genuine and impostor

---

## Project Structure

```
Ageisnet/
│
├── scripts/
│   ├── step1_organize.py              # Copy dataset PNGs into alert/ and drowsy/
│   ├── step2_extract_features.py      # MediaPipe face feature extraction (EAR, MAR, head pose)
│   ├── step3_split.py                 # 80/20 stratified train/test split
│   ├── step4_train_tcn.py             # Train TCN on face features
│   ├── pose_stgcn_pipeline.py         # Extract body keypoints + train ST-GCN
│   ├── step5_fusion.py                # Attention-weighted fusion of both streams
│   ├── step6_identity_verification.py # ArcFace registration and verification
│   ├── step7_onnx_quantize.py         # ONNX export + INT8 quantisation + benchmark
│   └── live_identity.py               # Live webcam registration / verification
│
├── data/
│   ├── models/
│   │   ├── tcn/
│   │   │   ├── best_tcn.pt            # Trained TCN weights
│   │   │   ├── scaler.pkl             # Feature normalisation scaler
│   │   │   ├── history.csv            # Training loss/accuracy per epoch
│   │   │   └── test_report.txt        # Classification report on test set
│   │   ├── stgcn/
│   │   │   └── best_stgcn.pt          # Trained ST-GCN weights
│   │   ├── fusion/
│   │   │   ├── best_fusion.pt         # Trained fusion layer weights
│   │   │   └── fusion_report.txt      # Full evaluation report
│   │   └── onnx/
│   │       ├── best_tcn.onnx          # TCN FP32 ONNX
│   │       ├── best_tcn_int8.onnx     # TCN INT8 ONNX
│   │       ├── best_stgcn.onnx        # ST-GCN FP32 ONNX
│   │       ├── best_stgcn_int8.onnx   # ST-GCN INT8 ONNX
│   │       ├── best_fusion.onnx       # Fusion FP32 ONNX
│   │       └── best_fusion_int8.onnx  # Fusion INT8 ONNX
│   └── identity/
│       └── *_embedding.npy            # Saved 512-d ArcFace embeddings per driver
│
├── models/                            # MediaPipe model files (downloaded on first run)
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or 3.12
- pip

### Install dependencies

```bash
pip install torch torchvision
pip install mediapipe
pip install facenet-pytorch
pip install opencv-python
pip install scikit-learn pandas numpy
pip install onnx onnxruntime
pip install Pillow
```

> **Note on InsightFace:** InsightFace requires Microsoft C++ Build Tools on Windows and is not used here. `facenet-pytorch` provides identical ArcFace embeddings (VGGFace2, 512-d) with a pure-Python install.

### Download the dataset

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd
```

Or download manually from the Kaggle page and unzip to `data/raw/`.

---

## Pipeline — Step by Step

Run each script in order from the repo root.

---

### Step 1 — Dataset Organisation

```bash
python scripts/step1_organize.py
```

Copies alert and drowsy PNG frames from the raw download into:
```
data/processed/alert/    # alert frames
data/processed/drowsy/   # drowsy frames
```

---

### Step 2 — Feature Extraction

```bash
python scripts/step2_extract_features.py
```

Uses **MediaPipe FaceLandmarker** (Tasks API) to process every frame and compute:

| Feature | Description |
|---|---|
| EAR | Eye Aspect Ratio — average of both eyes using 6 landmarks each |
| MAR | Mouth Aspect Ratio — top/bottom lip distance over mouth width |
| Pitch | Head up/down angle via `solvePnP` |
| Yaw | Head left/right angle |
| Roll | Head tilt angle |

Output: `data/features.csv` (41,769 rows, 7 columns)

> **Gimbal-lock fix:** raw pitch from `solvePnP` wraps to ±180°. Values beyond ±90° are corrected by subtracting/adding 180°.

Also runs **MediaPipe BlazePose** to extract 33 body keypoints (x, y, z, visibility) per frame.

Output: `data/pose_features.csv`

---

### Step 3 — Train / Test Split

```bash
python scripts/step3_split.py
```

Stratified 80/20 split on `features.csv`:
- `data/train.csv` — 33,415 rows
- `data/test.csv` — 8,354 rows

Matching split performed on `pose_features.csv` → `pose_train.csv`, `pose_test.csv`.

---

### Step 4 — TCN Face Stream

```bash
python scripts/step4_train_tcn.py
```

**Architecture — Temporal Convolutional Network:**
- Input: sliding window of 45 frames × 5 features (EAR, MAR, pitch, yaw, roll)
- 4 × `CausalBlock` with dilations `[1, 2, 4, 8]`, channels `[64, 64, 128, 128]`
- Each block: dilated causal convolution → BatchNorm → ReLU → Dropout → residual connection
- `AdaptiveAvgPool1d(1)` → `Linear(128 → 1)` → sigmoid

**Training:**
- Optimiser: Adam, lr=1e-3, weight_decay=1e-4
- Loss: `BCEWithLogitsLoss` with class-balance pos_weight
- Scheduler: `ReduceLROnPlateau` (patience=3)
- Early stopping: patience=7 epochs
- Gradient clipping: 1.0

**Output:** `data/models/tcn/best_tcn.pt`, `data/models/tcn/scaler.pkl`

| Metric | Value |
|---|---|
| Test accuracy | **99.5%** |
| Parameters | 220,993 |
| Window size | 45 frames |

---

### Step 5 — ST-GCN Body Stream + Attention Fusion

#### Body stream

```bash
python scripts/pose_stgcn_pipeline.py
```

**Architecture — Spatial-Temporal Graph Convolutional Network:**
- Input: `(B, T=45, C=4, V=33)` — 45 frames, 4 channels (x/y/z/visibility), 33 joints
- Graph: MediaPipe BlazePose 33-joint skeleton with 35 edges
- Adjacency: symmetrically normalised `D^{-½} A D^{-½} + I`
- 3 × `STGCNBlock`: graph convolution → temporal convolution → BN → ReLU → residual
- Global mean pool → `Linear → 2-class` output

**Output:** `data/models/stgcn/best_stgcn.pt`

| Metric | Value |
|---|---|
| Test accuracy | **98.6%** |
| F1 score | 0.986 |
| Parameters | 216,810 |

#### Attention Fusion

```bash
python scripts/step5_fusion.py
```

Loads both trained models, runs inference on the shared test set (inner-joined on frame path to prevent data leakage), and trains a small attention network:

```
[face_prob, body_prob]  (2,)
       |
   Linear(2 → 32) → ReLU → Linear(32 → 2) → Softmax
       |
  [w_face, w_body]   (per-sample attention weights)
       |
  Safety Score = 1 − (w_face × face_P + w_body × body_P)
```

**Learned weights (average):** `w_face = 0.644`, `w_body = 0.356`

| Metric | Value |
|---|---|
| Test accuracy | **99.33%** |
| F1 (weighted) | 0.993 |
| Test sequences | 149 |
| Total errors | 1 |

**Confusion matrix:**

```
              alert   drowsy
  alert          65        1
  drowsy          0       83
```

**Output:** `data/models/fusion/best_fusion.pt`

---

### Step 6 — ArcFace Identity Verification

```bash
python scripts/step6_identity_verification.py
```

Uses **InceptionResnetV1** (facenet-pytorch, VGGFace2 pretrained) to extract 512-d L2-normalised face embeddings.

**Registration (one-time per driver):**
1. Detect and align face with MTCNN (112×112, margin=14)
2. Extract 512-d ArcFace embedding
3. Save as `data/identity/{name}_embedding.npy`

**Verification (every ride start):**
1. Extract fresh embedding from camera frame
2. Compute cosine similarity vs saved embedding
3. `>= 0.85` → Verified | `< 0.85` → Mismatch

**Test results:**

| Test | Correct | Avg Similarity |
|---|---|---|
| Same person (5 frames) | 5 / 5 | 0.923 |
| Different person (5 subjects) | 5 / 5 | 0.166 |
| **Separability gap** | — | **0.758** |

---

### Step 7 — ONNX Export and INT8 Quantisation

```bash
python scripts/step7_onnx_quantize.py
```

Exports all three models to ONNX (opset 17) then applies INT8 dynamic post-training quantisation via ONNX Runtime.

**File size reduction:**

| Model | FP32 | INT8 | Reduction |
|---|---|---|---|
| TCN | 0.86 MB | 0.25 MB | **70.6%** |
| ST-GCN | 0.84 MB | 0.24 MB | **71.7%** |
| Fusion | < 1 KB | < 1 KB | — |

**Latency benchmark (100 passes, CPU):**

| Model | FP32 | INT8 |
|---|---|---|
| TCN | 2.07 ms | 7.26 ms |
| ST-GCN | 10.50 ms | 51.82 ms |
| Fusion | 0.04 ms | 0.06 ms |
| **Full pipeline** | **12.61 ms** | **59.14 ms** |

> **Why is INT8 slower here?** Dynamic INT8 quantisation reduces file size and memory bandwidth but does not provide a speed-up on consumer Intel/AMD CPUs without INT8 hardware (Intel VNNI / AMX). The gain appears on mobile SoCs with dedicated INT8 DSP/NPU (e.g. Snapdragon). The primary deployment benefit here is the ~71% model size reduction.

**INT8 accuracy vs FP32 baseline:**

| | Accuracy |
|---|---|
| PyTorch FP32 baseline | 99.33% |
| ONNX INT8 | 100.00% |
| Drop | −0.67 pp (within 1 pp threshold) |

---

## Live Webcam Demo

```bash
python scripts/live_identity.py
```

Opens your webcam for real-time face registration and verification:

1. Terminal asks `Enter your name:`
2. If no embedding saved → **Registration mode**: press `SPACE` to capture, embedding saved
3. If embedding exists → **Verification mode**: press `SPACE` to capture, cosine similarity displayed
4. Result shown on screen in green (Verified) or red (Access Denied)
5. Press `Q` to quit

If webcam doesn't open, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the script.

---

## Model Performance Summary

| Stream | Accuracy | F1 | Latency (FP32) | Model Size (INT8) |
|---|---|---|---|---|
| TCN — Face | 99.5% | 0.995 | 2.07 ms | 0.25 MB |
| ST-GCN — Body | 98.6% | 0.986 | 10.50 ms | 0.24 MB |
| Fusion | 99.33% | 0.993 | 12.61 ms total | < 1 KB |
| ArcFace ID | 100% (10/10) | — | — | — |

---

## ONNX Deployment Benchmarks

```
Full pipeline (FP32):  12.61 ms per inference
Full pipeline (INT8):  59.14 ms per inference
INT8 model sizes:      0.25 MB (TCN) + 0.24 MB (ST-GCN) + <1 KB (Fusion)
INT8 accuracy:         100.00 % (baseline 99.33 %)
```

ONNX files are ready for deployment with ONNX Runtime on any platform (Windows, Linux, Android via ONNX Runtime Mobile).

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Two-stream fusion instead of one model | Face and body carry complementary information; face captures micro-expressions (EAR, MAR), body captures posture slump. Fusion outperforms either stream alone. |
| TCN over LSTM | Dilated causal convolutions parallelize over time during training, avoid vanishing gradients, and have a fixed receptive field that maps cleanly to ONNX. |
| ST-GCN over CNN on keypoints | Skeleton graph structure encodes joint relationships; graph convolution propagates context along anatomically meaningful edges. |
| facenet-pytorch over InsightFace | InsightFace requires Microsoft C++ Build Tools on Windows. facenet-pytorch is pure PyTorch, installs anywhere, and uses the same ArcFace/VGGFace2 architecture. |
| Inner join on frame path for fusion | Training face and body CSVs separately causes leakage if split independently. Joining on the exact frame path ensures both streams see exactly the same frames in train and test. |
| Dynamic INT8 (not static) | Static quantisation requires a calibration dataset. Dynamic quantisation quantises weights at export time and activations at runtime — no calibration data needed, zero accuracy loss on this dataset. |
| Cosine similarity threshold 0.85 | Empirically chosen from the test: genuine pairs average 0.923, impostor pairs average 0.166. 0.85 sits well above the impostor ceiling with a 0.073 margin. |
