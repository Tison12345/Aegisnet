"""
Step 6: ArcFace Identity Verification — Phase 0 of the Ageisnet system.

Library: facenet-pytorch  (InceptionResnetV1 trained with ArcFace loss on
         VGGFace2, 512-d L2-normalised embeddings — identical in design to
         InsightFace's ArcFace model; installs without C++ build tools)

Registration (one-time per driver):
  - Detect and align face via MTCNN
  - Extract 512-d ArcFace embedding
  - Save as  data/identity/<name>_embedding.npy

Verification (every ride start):
  - Extract fresh embedding from new image
  - Cosine similarity vs stored embedding
  - >= 0.85  -->  Verified
  -  < 0.85  -->  Mismatch

Test:
  Dataset uses letter-prefixed filenames (a####.png = subject A, b####.png = B …).
  We register subject 'a', then verify:
    - 5 same-person images  (other 'a' frames) --> expect Verified
    - 5 different-person images (subjects b/c/d/e/f) --> expect Mismatch
"""

from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# ── config ───────────────────────────────────────────────────────────────────────
THRESHOLD    = 0.85
EMBED_DIR    = Path(r"C:\Ageisnet\data\identity")
PROCESSED    = Path(r"C:\Ageisnet\data\processed")
DEVICE       = torch.device("cpu")

# ── model init (downloaded once, cached in ~/.cache/torch) ──────────────────────
print("Loading MTCNN face detector and ArcFace model...")
mtcnn   = MTCNN(image_size=112, margin=14, device=DEVICE, keep_all=False, post_process=True)
arcface = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
print("Models ready.\n")


# ── core functions ───────────────────────────────────────────────────────────────
def extract_embedding(img_path: Path) -> np.ndarray | None:
    """
    Detect face, align, run ArcFace → return L2-normalised 512-d embedding.
    Returns None if no face is detected.
    """
    img_pil = Image.open(img_path).convert("RGB")
    face_tensor = mtcnn(img_pil)           # (3, 112, 112) or None
    if face_tensor is None:
        return None
    with torch.no_grad():
        emb = arcface(face_tensor.unsqueeze(0).to(DEVICE))   # (1, 512)
    emb_np = emb.squeeze(0).cpu().numpy()
    emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)       # L2 normalise
    return emb_np


def register(img_path: Path, name: str) -> np.ndarray:
    """Extract embedding from img_path and save as <name>_embedding.npy."""
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    emb = extract_embedding(img_path)
    if emb is None:
        raise ValueError(f"No face detected in registration image: {img_path}")
    save_path = EMBED_DIR / f"{name}_embedding.npy"
    np.save(save_path, emb)
    print(f"Registered '{name}' from {img_path.name}  -->  {save_path}")
    return emb


def verify(img_path: Path, registered_name: str) -> tuple[str, float]:
    """
    Compare face in img_path against stored registration embedding.
    Returns (result, cosine_similarity).
    result is 'Verified' or 'Mismatch' or 'NoFace'.
    """
    emb_path = EMBED_DIR / f"{registered_name}_embedding.npy"
    stored   = np.load(emb_path)

    fresh = extract_embedding(img_path)
    if fresh is None:
        return "NoFace", 0.0

    similarity = float(np.dot(stored, fresh))        # both L2-normalised → cosine sim
    result = "Verified" if similarity >= THRESHOLD else "Mismatch"
    return result, similarity


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    return float(np.dot(a, b))


# ── test helpers ─────────────────────────────────────────────────────────────────
def pick_frames(subject_prefix: str, folder: Path, n: int,
                skip: int = 50) -> list[Path]:
    """
    Return n frames from <folder> whose stem starts with subject_prefix,
    skipping the first `skip` frames (registration uses frame 0).
    """
    all_frames = sorted(folder.glob(f"{subject_prefix}*.png"))
    candidates = all_frames[skip:]
    # Spread picks evenly across available frames for variety
    if len(candidates) < n:
        return candidates[:n]
    step = max(1, len(candidates) // n)
    return [candidates[i * step] for i in range(n)]


# ── main test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ALERT_DIR  = PROCESSED / "alert"
    DROWSY_DIR = PROCESSED / "drowsy"

    # ── Registration ────────────────────────────────────────────────────────────
    # Subject 'a' — use the first alert frame as the registration image
    reg_frames = sorted(ALERT_DIR.glob("a*.png"))
    if not reg_frames:
        raise FileNotFoundError("No 'a' prefix frames in alert folder.")

    reg_image = reg_frames[0]
    print("=" * 60)
    print("REGISTRATION")
    print("=" * 60)
    stored_emb = register(reg_image, name="subject_a")
    print()

    # ── Same-person verification (5 frames of subject 'a') ───────────────────────
    same_frames = pick_frames("a", ALERT_DIR, n=5, skip=1)
    # Also look in drowsy folder for more variety (same person, different state)
    same_frames_drowsy = pick_frames("A", DROWSY_DIR, n=2, skip=1)
    same_tests = (same_frames + same_frames_drowsy)[:5]

    print("=" * 60)
    print("SAME-PERSON VERIFICATION (expect: Verified)")
    print("=" * 60)
    same_results = []
    for img in same_tests:
        result, sim = verify(img, "subject_a")
        flag = "OK" if result == "Verified" else "WRONG"
        print(f"  [{flag}] {img.name:<20s}  similarity={sim:.4f}  -->  {result}")
        same_results.append((result, sim))

    # ── Different-person verification (5 frames from subjects b/c/d/e/f) ────────
    diff_subjects = ["b", "c", "d", "e", "f"]
    diff_tests = []
    for subj in diff_subjects:
        frames = sorted(ALERT_DIR.glob(f"{subj}*.png"))
        if frames:
            # Pick a frame from the middle of the sequence for variety
            diff_tests.append(frames[len(frames) // 2])

    diff_tests = diff_tests[:5]
    if len(diff_tests) < 5:
        # Fall back to drowsy folder if needed
        for subj in ["B", "C", "D", "E", "F"]:
            frames = sorted(DROWSY_DIR.glob(f"{subj}*.png"))
            if frames and len(diff_tests) < 5:
                diff_tests.append(frames[len(frames) // 2])

    print()
    print("=" * 60)
    print("DIFFERENT-PERSON VERIFICATION (expect: Mismatch)")
    print("=" * 60)
    diff_results = []
    for img in diff_tests:
        result, sim = verify(img, "subject_a")
        flag = "OK" if result == "Mismatch" else "WRONG"
        print(f"  [{flag}] {img.name:<20s}  similarity={sim:.4f}  -->  {result}")
        diff_results.append((result, sim))

    # ── Summary ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    same_correct  = sum(1 for r, _ in same_results  if r == "Verified")
    diff_correct  = sum(1 for r, _ in diff_results  if r == "Mismatch")
    avg_same_sim  = np.mean([s for _, s in same_results])
    avg_diff_sim  = np.mean([s for _, s in diff_results])

    print(f"  Threshold              : {THRESHOLD}")
    print(f"  Same-person  correct   : {same_correct}/{len(same_results)}"
          f"  (avg similarity = {avg_same_sim:.4f})")
    print(f"  Diff-person  correct   : {diff_correct}/{len(diff_results)}"
          f"  (avg similarity = {avg_diff_sim:.4f})")
    print(f"  Separability gap       : {avg_same_sim - avg_diff_sim:.4f}"
          f"  (larger = better)")

    overall = same_correct + diff_correct
    total   = len(same_results) + len(diff_results)
    print(f"  Overall accuracy       : {overall}/{total}  ({100*overall/total:.0f}%)")
    print()
    print(f"Embedding stored at: {EMBED_DIR / 'subject_a_embedding.npy'}")
