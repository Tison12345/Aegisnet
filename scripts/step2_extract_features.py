"""
Step 2: Extract facial landmarks and compute EAR, MAR, head pose for every frame.

Uses MediaPipe Tasks API (face_landmarker.task model) — compatible with mediapipe >= 0.10.
For each frame in data/processed/{alert,drowsy}/:
  - Run FaceLandmarker (478 landmarks including irises)
  - Compute EAR  (Eye Aspect Ratio)       -- drowsiness proxy
  - Compute MAR  (Mouth Aspect Ratio)     -- yawn proxy
  - Compute head pitch, yaw, roll         -- distraction proxy

Output: data/features.csv
Columns: frame_path, label, EAR, MAR, pitch, yaw, roll

CPU-only, no GPU required.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── paths ───────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:/Ageisnet/data/processed")
MODEL_PATH = Path(r"C:/Ageisnet/models/face_landmarker.task")
OUT_CSV    = Path(r"C:/Ageisnet/data/features.csv")
CLASSES    = ["alert", "drowsy"]

# ── landmark index sets (MediaPipe canonical 478-pt model) ───────────────────────
# 6-point EAR landmarks per eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Outer lip landmarks for MAR
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

# Indices for head pose (nose tip, chin, left-eye corner, right-eye corner,
# left-mouth corner, right-mouth corner)
POSE_IDX = [1, 152, 33, 263, 61, 291]

# Corresponding 3-D model points (mm)
MODEL_POINTS_3D = np.array([
    [ 0.0,    0.0,    0.0  ],   # nose tip
    [ 0.0,  -63.6,  -12.5 ],   # chin
    [-43.3,  32.7,  -26.0 ],   # left eye left corner
    [ 43.3,  32.7,  -26.0 ],   # right eye right corner
    [-28.9, -28.9,  -24.1 ],   # left mouth corner
    [ 28.9, -28.9,  -24.1 ],   # right mouth corner
], dtype=np.float64)


def make_landmarker():
    """Build a FaceLandmarker configured for still-image processing."""
    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
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


def ear(lm, indices, w, h):
    """Eye Aspect Ratio for one eye (6-point formula)."""
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mar(lm, w, h):
    """Mouth Aspect Ratio (vertical opening / horizontal width)."""
    top    = (lm[MOUTH_TOP].x    * w, lm[MOUTH_TOP].y    * h)
    bottom = (lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h)
    left   = (lm[MOUTH_LEFT].x   * w, lm[MOUTH_LEFT].y   * h)
    right  = (lm[MOUTH_RIGHT].x  * w, lm[MOUTH_RIGHT].y  * h)
    return math.dist(top, bottom) / (math.dist(left, right) + 1e-6)


def head_pose(lm, w, h):
    """Estimate pitch, yaw, roll in degrees via solvePnP."""
    img_pts = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in POSE_IDX], dtype=np.float64
    )
    focal = float(w)
    cam   = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist  = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, img_pts, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll  = math.degrees(math.atan2( rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll  = 0.0

    # Wrap pitch from near-±180 back to near-0 (gimbal-lock ambiguity fix).
    # A frontal driver face decomposes as ~170° instead of ~-10°; adding/
    # subtracting 180° recovers the physically correct small angle.
    if pitch > 90:
        pitch -= 180.0
    elif pitch < -90:
        pitch += 180.0

    return pitch, yaw, roll


def process_image(landmarker, img_path: Path):
    """Return (EAR, MAR, pitch, yaw, roll) or None if no face detected."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]  # list of NormalizedLandmark

    avg_ear  = (ear(lm, LEFT_EYE, w, h) + ear(lm, RIGHT_EYE, w, h)) / 2.0
    mouth_ar = mar(lm, w, h)
    pitch, yaw, roll = head_pose(lm, w, h)

    if pitch is None:
        return None
    return avg_ear, mouth_ar, pitch, yaw, roll


# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    landmarker = make_landmarker()
    rows = []

    for label in CLASSES:
        folder = DATA_DIR / label
        images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        print(f"\nProcessing [{label}] -- {len(images)} images...")
        skipped = 0

        for i, img_path in enumerate(images):
            result = process_image(landmarker, img_path)
            if result is None:
                skipped += 1
                continue
            avg_ear, mouth_ar, pitch, yaw, roll = result
            rows.append({
                "frame_path": str(img_path),
                "label":      label,
                "EAR":        round(avg_ear,  4),
                "MAR":        round(mouth_ar, 4),
                "pitch":      round(pitch,    4),
                "yaw":        round(yaw,      4),
                "roll":       round(roll,     4),
            })
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(images)}  skipped={skipped}")

        print(f"  Done. skipped {skipped}/{len(images)} (no face detected).")

    landmarker.close()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_path", "label", "EAR", "MAR", "pitch", "yaw", "roll"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {OUT_CSV}")
