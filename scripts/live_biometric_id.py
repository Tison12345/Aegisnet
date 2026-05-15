"""
Behavioral biometric identity verification using TCN + ST-GCN penultimate embeddings.

Both models were trained to classify alert vs drowsy, and in doing so learned
person-discriminating representations: the face stream captures EAR/MAR/head-pose
dynamics, the body stream captures skeletal motion patterns. The 128-d penultimate
vector from each (after global pooling, before the classifier head) is concatenated
into a 256-d biometric vector that is unique to each person.

Registration mode  (first run, no profile saved):
  - Capture 45 frames from webcam
  - Extract 256-d biometric embedding via TCN + ST-GCN penultimate layers
  - Save as data/identity/{name}_biometric.npy

Verification mode  (profile already exists):
  - Capture 45 fresh frames
  - Extract 256-d embedding
  - Cosine similarity >= 0.80  ->  Verified
  - Cosine similarity <  0.80  ->  Access Denied

Controls: Q to quit at any time.
"""

import math
import pickle
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(r"C:\Ageisnet")
IDENTITY_DIR  = ROOT / "data" / "identity"
TCN_PT        = ROOT / "data" / "models" / "tcn" / "best_tcn.pt"
TCN_SCALER    = ROOT / "data" / "models" / "tcn" / "scaler.pkl"
STGCN_PT      = ROOT / "data" / "models" / "stgcn" / "best_stgcn.pt"
TASK_MODEL    = ROOT / "models" / "face_landmarker.task"
TASK_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ── hyper-params (must match training scripts) ─────────────────────────────────
N_FRAMES         = 45
THRESHOLD        = 0.80
DEVICE           = torch.device("cpu")
ROLL_CLIP        = 90.0
TCN_CHANNELS     = [64, 64, 128, 128]
TCN_KERNEL       = 3
TCN_DROPOUT      = 0.2
N_LANDMARKS      = 33
FEATS_PER_LM     = 4        # x, y, z, visibility
FACE_FEATURES    = ["EAR", "MAR", "pitch", "yaw", "roll"]

# ── face landmark indices (MediaPipe 478-pt model) ─────────────────────────────
LEFT_EYE    = [362, 385, 387, 263, 373, 380]
RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
MOUTH_TOP   = 13
MOUTH_BOT   = 14
MOUTH_LEFT  = 78
MOUTH_RIGHT = 308
POSE_IDX    = [1, 152, 33, 263, 61, 291]
MODEL_PTS   = np.array([
    [ 0.0,    0.0,    0.0  ],
    [ 0.0,  -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [ 43.3,  32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [ 28.9, -28.9,  -24.1 ],
], dtype=np.float64)

# ── ST-GCN skeleton edges (MediaPipe BlazePose 33-joint topology) ──────────────
POSE_EDGES = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(3,7),(6,8),
    (0,9),(0,10),(9,10),(0,11),(0,12),(11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]
EDGE_LIST = POSE_EDGES + [(j, i) for i, j in POSE_EDGES]

# ── display colours (BGR) ──────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
YELLOW = (0, 200, 255)
CYAN   = (255, 200, 0)


# ══════════════════════════════════════════════════════════════════════════════
# TCN — must match step4_train_tcn.py exactly so state_dict loads correctly
# ══════════════════════════════════════════════════════════════════════════════
class CausalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch
                           else nn.Identity())
        self.pad = pad

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)[:, :, :-self.pad or None]))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)[:, :, :-self.pad or None]))
        out = self.drop(out)
        return self.relu(out + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, in_ch, channels, kernel, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(CausalBlock(
                in_ch if i == 0 else channels[i - 1],
                out_ch, kernel, dilation, dropout,
            ))
        self.net  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):
        out = self.net(x)
        out = self.pool(out)
        return self.head(out.squeeze(-1))

    @torch.no_grad()
    def embed(self, x):
        """Penultimate 128-d embedding (before classification head)."""
        out = self.net(x)
        out = self.pool(out)
        return out.squeeze(-1)          # (B, 128)


# ══════════════════════════════════════════════════════════════════════════════
# ST-GCN — must match pose_stgcn_pipeline.py exactly
# ══════════════════════════════════════════════════════════════════════════════
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A    = nn.Parameter(A, requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.einsum("nctv,vw->nctw", x, self.A)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, dropout=0.3):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        pad      = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(kernel_size, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                          nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else nn.Sequential()
        )

    def forward(self, x):
        res = self.residual(x)
        x   = self.gcn(x)
        x   = self.tcn(x)
        return nn.functional.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, in_channels, num_classes, A):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * N_LANDMARKS)
        self.layer1  = STGCNBlock(in_channels, 32,  A)
        self.layer2  = STGCNBlock(32,          64,  A)
        self.layer3  = STGCNBlock(64,          128, A)
        self.fc      = nn.Linear(128, num_classes)

    def _backbone(self, x):
        """Shared feature extraction up to penultimate pooling."""
        N, T, C, V = x.shape
        x = x.permute(0, 2, 1, 3)          # (N, C, T, V)
        x = x.reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)   # (N, C, T, V) -> (N, C, V, T) -> wait
        # Note: after reshape(N, C, V, T) we get (N,C,V,T), then permute(0,1,3,2) -> (N,C,T,V)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)                  # (N, 128, T, V)
        return x.mean(dim=2).mean(dim=2)    # (N, 128)

    def forward(self, x):
        return self.fc(self._backbone(x))

    @torch.no_grad()
    def embed(self, x):
        """Penultimate 128-d embedding (before classification head)."""
        return self._backbone(x)            # (B, 128)


# ══════════════════════════════════════════════════════════════════════════════
# Adjacency matrix helper
# ══════════════════════════════════════════════════════════════════════════════
def make_adjacency_matrix() -> torch.Tensor:
    A = np.zeros((N_LANDMARKS, N_LANDMARKS), dtype=np.float32)
    for i, j in EDGE_LIST:
        A[i, j] = 1.0
    A += np.eye(N_LANDMARKS, dtype=np.float32)
    D      = A.sum(axis=1)
    D_inv  = np.diag(1.0 / np.sqrt(np.where(D > 0, D, 1.0)))
    return torch.tensor(D_inv @ A @ D_inv, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
def load_tcn() -> TCN:
    model = TCN(len(FACE_FEATURES), TCN_CHANNELS, TCN_KERNEL, TCN_DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(str(TCN_PT), map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    return model


def load_stgcn() -> STGCN:
    A     = make_adjacency_matrix().to(DEVICE)
    model = STGCN(in_channels=FEATS_PER_LM, num_classes=2, A=A).to(DEVICE)
    model.load_state_dict(torch.load(str(STGCN_PT), map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    return model


def load_scaler():
    with open(TCN_SCALER, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe helpers
# ══════════════════════════════════════════════════════════════════════════════
def ensure_task_model():
    """Download face_landmarker.task if not present."""
    if TASK_MODEL.exists():
        return
    print(f"Downloading face_landmarker.task to {TASK_MODEL} ...")
    TASK_MODEL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_URL, str(TASK_MODEL))
    print("  Download complete.")


def make_face_landmarker():
    ensure_task_model()
    base_opts = mp_python.BaseOptions(model_asset_path=str(TASK_MODEL))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def make_pose_detector():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )


# ── per-frame feature extractors ───────────────────────────────────────────────
def _ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(lm, w, h):
    top    = (lm[MOUTH_TOP].x   * w, lm[MOUTH_TOP].y   * h)
    bot    = (lm[MOUTH_BOT].x   * w, lm[MOUTH_BOT].y   * h)
    left   = (lm[MOUTH_LEFT].x  * w, lm[MOUTH_LEFT].y  * h)
    right  = (lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h)
    return math.dist(top, bot) / (math.dist(left, right) + 1e-6)


def _head_pose(lm, w, h):
    img_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in POSE_IDX],
                       dtype=np.float64)
    focal = float(w)
    cam   = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_PTS, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( rmat[2,1], rmat[2,2]))
        yaw   = math.degrees(math.atan2(-rmat[2,0], sy))
        roll  = math.degrees(math.atan2( rmat[1,0], rmat[0,0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1,2], rmat[1,1]))
        yaw   = math.degrees(math.atan2(-rmat[2,0], sy))
        roll  = 0.0
    if pitch >  90: pitch -= 180.0
    if pitch < -90: pitch += 180.0
    return pitch, yaw, roll


def extract_face_features(bgr_frame, landmarker) -> np.ndarray | None:
    """Return [EAR, MAR, pitch, yaw, roll] or None if no face detected."""
    h, w = bgr_frame.shape[:2]
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lm    = result.face_landmarks[0]
    pitch, yaw, roll = _head_pose(lm, w, h)
    if pitch is None:
        return None
    avg_ear = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2.0
    mouth_ar = _mar(lm, w, h)
    return np.array([avg_ear, mouth_ar, pitch, yaw, roll], dtype=np.float64)


def extract_pose_keypoints(bgr_frame, pose_det) -> np.ndarray | None:
    """Return flat (132,) array of [x,y,z,vis] * 33 landmarks, or None."""
    rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    result = pose_det.process(rgb)
    if not result.pose_landmarks:
        return None
    row = []
    for lm in result.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# OSD helpers
# ══════════════════════════════════════════════════════════════════════════════
def put_text(frame, text, pos, colour, scale=0.7, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour,  thickness,   cv2.LINE_AA)


def draw_progress_bar(frame, filled, total, y=30):
    h, w = frame.shape[:2]
    bar_w = w - 40
    cv2.rectangle(frame, (20, y), (20 + bar_w, y + 18), (60,60,60), -1)
    filled_w = int(bar_w * filled / total)
    cv2.rectangle(frame, (20, y), (20 + filled_w, y + 18), CYAN, -1)
    cv2.rectangle(frame, (20, y), (20 + bar_w, y + 18), WHITE, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Sequence capture
# ══════════════════════════════════════════════════════════════════════════════
def capture_sequence(mode_label: str, name: str) -> list:
    """
    Open webcam, capture N_FRAMES consecutive frames, show countdown bar.
    Returns list of BGR frames, or empty list if user quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open any webcam (tried index 0 and 1).")

    frames = []
    print(f"\nCapturing {N_FRAMES} frames for {mode_label.upper()} of '{name}' ...")
    print("Camera window open. Recording starts immediately. Press Q to abort.\n")

    while len(frames) < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        n_done  = len(frames)

        draw_progress_bar(display, n_done, N_FRAMES, y=10)
        put_text(display, f"{mode_label.upper()} — {name}",
                 (20, 65), YELLOW, scale=0.75)
        put_text(display, f"Capturing frame {n_done+1} / {N_FRAMES}",
                 (20, 95), WHITE, scale=0.6)
        put_text(display, f"Frames remaining: {N_FRAMES - n_done}",
                 (20, 120), CYAN, scale=0.6)

        cv2.imshow("Ageisnet — Biometric Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return []

        frames.append(frame.copy())

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {len(frames)} frames.")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Feature sequence builders
# ══════════════════════════════════════════════════════════════════════════════
def build_face_sequence(frames, landmarker, scaler) -> np.ndarray:
    """
    Returns (N_FRAMES, 5) float32 array scaled with the TCN scaler.
    Frames where no face is detected are filled with zeros (before scaling).
    """
    ZERO = np.zeros(len(FACE_FEATURES), dtype=np.float64)
    rows, n_fail = [], 0
    for i, frame in enumerate(frames):
        feat = extract_face_features(frame, landmarker)
        if feat is None:
            rows.append(ZERO.copy())
            n_fail += 1
        else:
            rows.append(feat)
    if n_fail:
        print(f"  [face] No face detected in {n_fail}/{len(frames)} frames "
              f"(zeros substituted).")
    seq = np.array(rows, dtype=np.float32)        # (N_FRAMES, 5)
    seq[:, 4] = np.clip(seq[:, 4], -ROLL_CLIP, ROLL_CLIP)   # clip roll
    seq = scaler.transform(seq).astype(np.float32)
    return seq


def build_pose_sequence(frames, pose_det) -> np.ndarray:
    """
    Returns (N_FRAMES, N_LANDMARKS, FEATS_PER_LM) float32 array.
    Frames where no body is detected are zero-filled.
    """
    ZERO = np.zeros(N_LANDMARKS * FEATS_PER_LM, dtype=np.float32)
    rows, n_fail = [], 0
    for frame in frames:
        kp = extract_pose_keypoints(frame, pose_det)
        if kp is None:
            rows.append(ZERO.copy())
            n_fail += 1
        else:
            rows.append(kp)
    if n_fail:
        print(f"  [pose] No body detected in {n_fail}/{len(frames)} frames "
              f"(zeros substituted).")
    seq = np.array(rows, dtype=np.float32)                        # (N_FRAMES, 132)
    return seq.reshape(N_FRAMES, N_LANDMARKS, FEATS_PER_LM)       # (45, 33, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ══════════════════════════════════════════════════════════════════════════════
def get_tcn_embedding(model, face_seq: np.ndarray) -> np.ndarray:
    """
    face_seq: (N_FRAMES, 5) float32
    Returns 128-d L2-normalised embedding.
    """
    x = torch.tensor(face_seq).unsqueeze(0)     # (1, 45, 5)
    x = x.permute(0, 2, 1)                      # (1, 5, 45) — Conv1d: (B, F, W)
    emb = model.embed(x).squeeze(0).numpy()     # (128,)
    return emb / (np.linalg.norm(emb) + 1e-8)


def get_stgcn_embedding(model, pose_seq: np.ndarray) -> np.ndarray:
    """
    pose_seq: (N_FRAMES, N_LANDMARKS, FEATS_PER_LM) float32
    Returns 128-d L2-normalised embedding.
    """
    x = torch.tensor(pose_seq)                  # (45, 33, 4) = (T, V, C)
    x = x.permute(0, 2, 1)                      # (45, 4, 33) = (T, C, V) — matches __getitem__
    x = x.unsqueeze(0)                          # (1, 45, 4, 33) = (B, T, C, V)
    emb = model.embed(x).squeeze(0).numpy()     # (128,)
    return emb / (np.linalg.norm(emb) + 1e-8)


def build_unified_embedding(tcn_model, stgcn_model,
                            face_seq, pose_seq) -> np.ndarray:
    """Concatenate TCN + ST-GCN penultimate embeddings -> 256-d vector."""
    emb_face = get_tcn_embedding(tcn_model, face_seq)
    emb_body = get_stgcn_embedding(stgcn_model, pose_seq)
    unified  = np.concatenate([emb_face, emb_body])         # (256,)
    return unified / (np.linalg.norm(unified) + 1e-8)       # L2-normalise


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ══════════════════════════════════════════════════════════════════════════════
# Result display
# ══════════════════════════════════════════════════════════════════════════════
def show_result_window(lines: list, hold_seconds: float = 4.0):
    """
    Show a plain dark window with result text for `hold_seconds`.
    lines: list of (text, colour) tuples.
    """
    blank = np.zeros((280, 560, 3), dtype=np.uint8)
    y = 50
    for text, colour in lines:
        put_text(blank, text, (30, y), colour, scale=0.8, thickness=2)
        y += 50
    put_text(blank, "Window closes automatically ...", (30, y + 10),
             (120, 120, 120), scale=0.5)
    cv2.imshow("Ageisnet — Result", blank)
    ms = max(1, int(hold_seconds * 1000))
    # Poll Q to allow early exit
    deadline = cv2.getTickCount() + int(hold_seconds * cv2.getTickFrequency())
    while cv2.getTickCount() < deadline:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    name = input("Enter your name: ").strip()
    if not name:
        sys.exit("Name cannot be empty.")

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = IDENTITY_DIR / f"{name}_biometric.npy"
    mode         = "verify" if profile_path.exists() else "register"

    print(f"\nMode : {'VERIFICATION' if mode == 'verify' else 'REGISTRATION'}")
    print(f"Name : {name}")
    print(f"Profile: {profile_path}")

    # ── load models ────────────────────────────────────────────────────────────
    print("\nLoading models ...")
    tcn_model   = load_tcn()
    stgcn_model = load_stgcn()
    scaler      = load_scaler()
    print("  TCN, ST-GCN, scaler — ready.")

    print("Loading MediaPipe detectors ...")
    landmarker = make_face_landmarker()
    pose_det   = make_pose_detector()
    print("  Face + pose detectors — ready.\n")

    # ── capture ────────────────────────────────────────────────────────────────
    frames = capture_sequence(mode, name)
    if not frames:
        print("Aborted.")
        sys.exit(0)

    # ── extract sequences ──────────────────────────────────────────────────────
    print("\nExtracting features ...")
    face_seq = build_face_sequence(frames, landmarker, scaler)
    pose_seq = build_pose_sequence(frames, pose_det)

    landmarker.close()
    pose_det.close()

    # ── compute unified 256-d embedding ────────────────────────────────────────
    print("Computing biometric embedding ...")
    embedding = build_unified_embedding(tcn_model, stgcn_model, face_seq, pose_seq)
    print(f"  Embedding shape: {embedding.shape}  norm: {np.linalg.norm(embedding):.4f}")

    # ── register ───────────────────────────────────────────────────────────────
    if mode == "register":
        np.save(str(profile_path), embedding)
        msg = f"Registered successfully -- {name}"
        print(f"\n{msg}")
        print(f"Profile saved: {profile_path}")
        show_result_window([
            (msg,                                      GREEN),
            (f"Profile: {profile_path.name}",          WHITE),
            ("Run again to verify.",                   YELLOW),
        ])
        return

    # ── verify ─────────────────────────────────────────────────────────────────
    stored = np.load(str(profile_path))
    sim    = cosine_sim(stored, embedding)
    sim_pct = sim * 100

    if sim >= THRESHOLD:
        verdict     = f"Verified -- Welcome, {name}!"
        verdict_col = GREEN
        outcome     = "VERIFIED"
    else:
        verdict     = "Mismatch -- Access Denied"
        verdict_col = RED
        outcome     = "DENIED"

    print(f"\n{'='*50}")
    print(f"  {outcome}")
    print(f"  Similarity  : {sim:.4f}  ({sim_pct:.1f} %)")
    print(f"  Threshold   : {THRESHOLD}")
    print(f"{'='*50}")

    show_result_window([
        (verdict,                                      verdict_col),
        (f"Similarity : {sim:.4f}  ({sim_pct:.1f} %)", WHITE),
        (f"Threshold  : {THRESHOLD}",                  WHITE),
        (f"Name       : {name}",                       YELLOW),
    ])


if __name__ == "__main__":
    main()
