"""
Live webcam identity registration and verification using ArcFace.

Usage:
  python live_identity.py

- No embedding saved  --> Registration mode: captures your face and saves embedding.
- Embedding exists    --> Verification mode:  compares live face against saved embedding.

Press SPACE to capture the live frame.
Press Q     to quit at any time.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# ── config ────────────────────────────────────────────────────────────────────
EMBED_DIR  = Path(r"C:\Ageisnet\data\identity")
THRESHOLD  = 0.85
DEVICE     = torch.device("cpu")

# colours (BGR)
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
YELLOW = (0, 200, 255)

# ── model init ────────────────────────────────────────────────────────────────
print("Loading ArcFace model and MTCNN detector …")
mtcnn   = MTCNN(image_size=112, margin=14, device=DEVICE,
                keep_all=False, post_process=True)
arcface = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
print("Models ready.\n")

EMBED_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def extract_embedding(bgr_frame: np.ndarray) -> np.ndarray | None:
    """BGR OpenCV frame -> L2-normalised 512-d ArcFace embedding, or None."""
    rgb_pil   = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    face_t    = mtcnn(rgb_pil)
    if face_t is None:
        return None
    with torch.no_grad():
        emb = arcface(face_t.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


def overlay_text(frame, lines, start_y=30, scale=0.7, thickness=2):
    """Draw a list of (text, colour) tuples onto frame."""
    y = start_y
    for text, colour in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # shadow
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thickness, cv2.LINE_AA)
        y += 34


# ── main ──────────────────────────────────────────────────────────────────────
name      = input("Enter your name: ").strip()
if not name:
    sys.exit("Name cannot be empty.")

embed_path = EMBED_DIR / f"{name}_embedding.npy"
mode       = "verify" if embed_path.exists() else "register"

print(f"\nMode : {'VERIFICATION' if mode == 'verify' else 'REGISTRATION'}")
print(f"Name : {name}")
if mode == "verify":
    print(f"Comparing against: {embed_path}")
print("\nPress SPACE to capture | Press Q to quit\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Cannot open webcam.")

result_text  = []   # list of (str, colour) shown after capture
result_ready = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if not result_ready:
        hint_colour = YELLOW if mode == "register" else (200, 200, 255)
        mode_label  = "REGISTRATION" if mode == "register" else "VERIFICATION"
        overlay_text(display, [
            (f"Mode: {mode_label}  |  Name: {name}", hint_colour),
            ("SPACE = capture    Q = quit",           WHITE),
        ])
    else:
        overlay_text(display, result_text)

    cv2.imshow("Ageisnet — Identity", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' ') and not result_ready:
        emb = extract_embedding(frame)

        if emb is None:
            result_text  = [("No face detected — try again", RED)]
            result_ready = False          # allow re-capture
            print("No face detected in frame — try again.")
            continue

        # ── REGISTRATION ──────────────────────────────────────────────────────
        if mode == "register":
            np.save(embed_path, emb)
            msg = f"Registered successfully -- {name}"
            print(f"\n{msg}")
            print(f"Embedding saved to: {embed_path}")
            result_text  = [(msg, GREEN),
                            ("Press Q to quit.", WHITE)]
            result_ready = True

        # ── VERIFICATION ─────────────────────────────────────────────────────
        else:
            stored = np.load(embed_path)
            sim    = float(np.dot(stored, emb))
            sim_pct = sim * 100

            if sim >= THRESHOLD:
                verdict      = f"Verified -- Welcome, {name}!"
                verdict_col  = GREEN
                print(f"\nVerified -- Welcome, {name}!  (similarity = {sim:.4f})")
            else:
                verdict      = "Mismatch -- Access Denied"
                verdict_col  = RED
                print(f"\nMismatch -- Access Denied  (similarity = {sim:.4f})")

            result_text = [
                (verdict,                         verdict_col),
                (f"Similarity: {sim:.4f}  ({sim_pct:.1f}%)", WHITE),
                (f"Threshold : {THRESHOLD}",      WHITE),
                ("Press Q to quit.",              YELLOW),
            ]
            result_ready = True

cap.release()
cv2.destroyAllWindows()
