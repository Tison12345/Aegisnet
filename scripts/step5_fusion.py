"""
Step 5: Attention-weighted fusion of Face TCN + Body ST-GCN streams.

Problem with independent splits
--------------------------------
face/pose train-test splits were made separately, so the same frame can land in
face_train but pose_test.  We solve this by merging both CSVs on their shared
frame path (normalised), keeping only frames that appear in the SAME split for
BOTH streams, then building sequences from that common subset.

Common frames available:  ~27 k train  /  ~2.3 k test  → ~1 800 / ~149 sequences.

Fusion design
-------------
Input:   [face_P(drowsy), body_P(drowsy)]  -- floats in [0, 1]
Attention MLP(2→16→2) + softmax → [w_face, w_body]  summing to 1 (per sample)
Safety score = 1 − (w_face · face_P + w_body · body_P)
  → 1.0 = fully alert, 0.0 = fully drowsy  (matches Phase 3 thresholds)
Classification threshold: safety_score < 0.5  ⟹  drowsy

Outputs
-------
  data/models/fusion/best_fusion.pt   -- fusion weights
  data/models/fusion/fusion_report.txt -- evaluation report
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── import trained model architectures ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from step4_train_tcn import (DROPOUT, FEATURES as FACE_FEATURES,
                              KERNEL_SIZE, TCN, TCN_CHANNELS)
from pose_stgcn_pipeline import (FEATURES_PER_LANDMARK, N_LANDMARKS,
                                  STGCN, make_adjacency_matrix)

# ── paths ───────────────────────────────────────────────────────────────────────
ROOT         = Path(r"C:\Ageisnet")
DATA         = ROOT / "data"
MODELS       = DATA / "models"
TCN_WEIGHTS  = MODELS / "tcn"   / "best_tcn.pt"
TCN_SCALER   = MODELS / "tcn"   / "scaler.pkl"
STGCN_WEIGHTS= MODELS / "stgcn" / "best_stgcn.pt"
FUSION_DIR   = MODELS / "fusion"

FACE_TRAIN   = DATA / "train.csv"
FACE_TEST    = DATA / "test.csv"
POSE_TRAIN   = DATA / "pose_train.csv"
POSE_TEST    = DATA / "pose_test.csv"

# ── hyper-params (must match both trained models) ───────────────────────────────
WINDOW     = 45
STRIDE     = 15
ROLL_CLIP  = 90.0
SEED       = 42
DEVICE     = torch.device("cpu")
BATCH      = 64
EPOCHS     = 20
LR         = 1e-3

POSE_COLS  = [f"v{i}" for i in range(N_LANDMARKS * FEATURES_PER_LANDMARK)]
LABEL_MAP  = {"alert": 0, "drowsy": 1}


# ── data helpers ─────────────────────────────────────────────────────────────────
def load_and_merge(face_csv: Path, pose_csv: Path) -> pd.DataFrame:
    """
    Inner-join face and pose DataFrames on normalised frame path.
    Returns a DataFrame with face features + pose features + label column.
    Only frames present in BOTH CSVs are kept.
    """
    face = pd.read_csv(face_csv)
    pose = pd.read_csv(pose_csv)

    face["_path"] = face["frame_path"].apply(lambda p: str(Path(p)))
    pose["_path"] = pose["image_path"].apply(lambda p: str(Path(p)))

    face["roll"] = face["roll"].clip(-ROLL_CLIP, ROLL_CLIP)

    merged = face.merge(
        pose.drop(columns=["image_path"]),
        on=["_path", "label"],
        how="inner",
    ).drop(columns=["frame_path", "_path"])
    return merged


def build_sequences(df: pd.DataFrame, face_scaler: StandardScaler,
                    pose_scaler: StandardScaler):
    """
    Build aligned face + pose sequences from a merged DataFrame.
    Returns X_face (N,W,F), X_pose (N,W,V,C), y (N,) all numpy.
    Sequences are built within each class to avoid label boundary crossings.
    """
    xf, xp, yl = [], [], []
    for label in ("alert", "drowsy"):
        sub = df[df["label"] == label].sort_values("_path" if "_path" in df else "label")
        # If _path was dropped from merged, sort by label is stable within class
        # Re-sort by the index which preserves per-class sort from merge
        face_vals = sub[FACE_FEATURES].values.astype(np.float32)
        pose_vals = sub[POSE_COLS].values.astype(np.float32)

        face_vals = face_scaler.transform(face_vals)
        pose_vals = pose_scaler.transform(pose_vals)

        n = len(sub)
        for start in range(0, n - WINDOW + 1, STRIDE):
            xf.append(face_vals[start: start + WINDOW])
            xp.append(pose_vals[start: start + WINDOW]
                      .reshape(WINDOW, N_LANDMARKS, FEATURES_PER_LANDMARK))
            yl.append(LABEL_MAP[label])

    return (np.stack(xf).astype(np.float32),
            np.stack(xp).astype(np.float32),
            np.array(yl, dtype=np.int64))


# ── model loaders ─────────────────────────────────────────────────────────────────
def load_tcn() -> TCN:
    m = TCN(len(FACE_FEATURES), TCN_CHANNELS, KERNEL_SIZE, DROPOUT)
    m.load_state_dict(torch.load(TCN_WEIGHTS, weights_only=True, map_location=DEVICE))
    m.eval()
    return m


def load_stgcn() -> STGCN:
    A = make_adjacency_matrix().to(DEVICE)
    m = STGCN(in_channels=FEATURES_PER_LANDMARK, num_classes=2, A=A)
    m.load_state_dict(torch.load(STGCN_WEIGHTS, weights_only=True, map_location=DEVICE))
    m.eval()
    return m


# ── inference ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def tcn_probs(model: TCN, X: np.ndarray) -> np.ndarray:
    """X: (N, W, F)  → P(drowsy) shape (N,)"""
    t = torch.from_numpy(X).permute(0, 2, 1)   # (N, F, W)
    logits = model(t)                            # (N, 1)
    return torch.sigmoid(logits).squeeze(1).numpy()


@torch.no_grad()
def stgcn_probs(model: STGCN, X: np.ndarray) -> np.ndarray:
    """X: (N, W, V, C)  → P(drowsy) shape (N,)"""
    # STGCN.forward expects (B, T, C, V)
    t = torch.from_numpy(X).permute(0, 1, 3, 2)  # (N, W, C, V)
    out = []
    for i in range(0, len(t), BATCH):
        logits = model(t[i: i + BATCH])           # (B, 2)
        out.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(out)


# ── fusion model ─────────────────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Learned attention over two stream probabilities.

    For each sample the MLP sees [face_P, body_P] and outputs
    soft attention weights [w_face, w_body] summing to 1.
    The weighted sum is then inverted to produce a safety score.
    """

    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 2) = [face_P_drowsy, body_P_drowsy]
        weights = torch.softmax(self.attn(x), dim=-1)       # (B, 2), sums to 1
        fused_P_drowsy = (weights * x).sum(dim=-1)           # (B,)
        safety_score   = 1.0 - fused_P_drowsy               # (B,)  1=alert, 0=drowsy
        return safety_score, weights


# ── evaluation helper ─────────────────────────────────────────────────────────────
def evaluate(model: AttentionFusion, loader: DataLoader):
    model.eval()
    all_scores, all_labels, all_weights = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            score, w = model(Xb.to(DEVICE))
            all_scores.extend(score.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_weights.extend(w.cpu().numpy())
    scores  = np.array(all_scores)
    labels  = np.array(all_labels).astype(int)
    preds   = (scores < 0.5).astype(int)    # safety_score < 0.5 → drowsy
    weights = np.array(all_weights)
    return scores, labels, preds, weights


# ── main ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    FUSION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & merge face + pose on common frames ──────────────────────────────
    print("Merging face and pose features on common frames...")
    train_merged = load_and_merge(FACE_TRAIN, POSE_TRAIN)
    test_merged  = load_and_merge(FACE_TEST,  POSE_TEST)
    print(f"  Train: {len(train_merged)} common frames  "
          f"({train_merged['label'].value_counts().to_dict()})")
    print(f"  Test:  {len(test_merged)}  common frames  "
          f"({test_merged['label'].value_counts().to_dict()})")

    # 2. Fit scalers on train ────────────────────────────────────────────────────
    with open(TCN_SCALER, "rb") as f:
        face_scaler = pickle.load(f)

    pose_scaler = StandardScaler()
    pose_scaler.fit(train_merged[POSE_COLS].values.astype(np.float32))

    # 3. Build aligned sequences ─────────────────────────────────────────────────
    print(f"\nBuilding sequences  window={WINDOW}  stride={STRIDE}...")
    train_merged = train_merged.sort_values(["label", "frame_path"]
                                            if "frame_path" in train_merged.columns
                                            else "label").reset_index(drop=True)
    test_merged  = test_merged.sort_values(["label", "frame_path"]
                                           if "frame_path" in test_merged.columns
                                           else "label").reset_index(drop=True)

    # Re-add _path column for sorting within build_sequences
    train_merged["_path"] = train_merged.index.astype(str)
    test_merged["_path"]  = test_merged.index.astype(str)

    Xf_tr, Xp_tr, y_tr = build_sequences(train_merged, face_scaler, pose_scaler)
    Xf_te, Xp_te, y_te = build_sequences(test_merged,  face_scaler, pose_scaler)

    print(f"  Train sequences: {len(y_tr)}  "
          f"(alert={int((y_tr==0).sum())}, drowsy={int((y_tr==1).sum())})")
    print(f"  Test  sequences: {len(y_te)}  "
          f"(alert={int((y_te==0).sum())}, drowsy={int((y_te==1).sum())})")

    # 4. Load models and run inference ───────────────────────────────────────────
    print("\nRunning TCN inference...")
    tcn   = load_tcn()
    fp_tr = tcn_probs(tcn, Xf_tr)
    fp_te = tcn_probs(tcn, Xf_te)

    print("Running ST-GCN inference...")
    stgcn = load_stgcn()
    bp_tr = stgcn_probs(stgcn, Xp_tr)
    bp_te = stgcn_probs(stgcn, Xp_te)

    # Individual model accuracy on the common-frame sequences
    face_tr_acc = accuracy_score(y_tr, (fp_tr >= 0.5).astype(int))
    body_tr_acc = accuracy_score(y_tr, (bp_tr >= 0.5).astype(int))
    face_te_acc = accuracy_score(y_te, (fp_te >= 0.5).astype(int))
    body_te_acc = accuracy_score(y_te, (bp_te >= 0.5).astype(int))
    print(f"\n  TCN  accuracy  -- train: {face_tr_acc:.4f}   test: {face_te_acc:.4f}")
    print(f"  STGCN accuracy -- train: {body_tr_acc:.4f}   test: {body_te_acc:.4f}")

    # 5. Build fusion datasets ────────────────────────────────────────────────────
    X_fus_tr = torch.tensor(np.stack([fp_tr, bp_tr], axis=1), dtype=torch.float32)
    X_fus_te = torch.tensor(np.stack([fp_te, bp_te], axis=1), dtype=torch.float32)
    y_fus_tr = torch.tensor(y_tr, dtype=torch.float32)
    y_fus_te = torch.tensor(y_te, dtype=torch.float32)

    train_ds = TensorDataset(X_fus_tr, y_fus_tr)
    test_ds  = TensorDataset(X_fus_te, y_fus_te)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    # 6. Train fusion layer ───────────────────────────────────────────────────────
    print("\nTraining attention fusion layer...")
    fusion    = AttentionFusion().to(DEVICE)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=LR, weight_decay=1e-4)
    # BCELoss on safety_score vs (1 − label):  alert → target=1, drowsy → target=0
    criterion = nn.BCELoss()

    best_acc     = 0.0
    patience_ctr = 0
    PATIENCE     = 7

    for epoch in range(1, EPOCHS + 1):
        fusion.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            safety_score, _ = fusion(Xb)
            loss = criterion(safety_score, 1.0 - yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        _, _, preds_val, _ = evaluate(fusion, test_loader)
        val_acc = accuracy_score(y_te, preds_val)

        tag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            patience_ctr = 0
            torch.save(fusion.state_dict(), FUSION_DIR / "best_fusion.pt")
            tag = " <- best"
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

        print(f"  Epoch {epoch:02d}/{EPOCHS}  loss={epoch_loss/len(train_loader):.4f}"
              f"  val_acc={val_acc:.4f}{tag}")

    # 7. Final evaluation ─────────────────────────────────────────────────────────
    fusion.load_state_dict(torch.load(FUSION_DIR / "best_fusion.pt",
                                      weights_only=True))
    scores, labels, preds, weights = evaluate(fusion, test_loader)

    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="weighted")
    cm     = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["alert", "drowsy"])

    avg_w_face = weights[:, 0].mean()
    avg_w_body = weights[:, 1].mean()

    result = (
        "\n" + "=" * 55 + "\n"
        "=== FUSION TEST RESULTS ===\n"
        "=" * 55 + "\n\n"
        f"Sequences evaluated: {len(labels)}\n"
        f"  (alert={int((labels==0).sum())}, drowsy={int((labels==1).sum())})\n\n"
        f"Individual stream accuracy on these sequences:\n"
        f"  Face TCN  : {face_te_acc:.4f}\n"
        f"  Body STGCN: {body_te_acc:.4f}\n\n"
        f"Fused model accuracy : {acc:.4f}\n"
        f"Fused model F1 (wtd) : {f1:.4f}\n\n"
        f"Avg learned attention weights:\n"
        f"  w_face = {avg_w_face:.3f}   w_body = {avg_w_body:.3f}  (sum={avg_w_face+avg_w_body:.3f})\n\n"
        f"Confusion matrix (rows=actual, cols=predicted):\n"
        f"             alert  drowsy\n"
        f"  alert       {cm[0,0]:4d}    {cm[0,1]:4d}\n"
        f"  drowsy      {cm[1,0]:4d}    {cm[1,1]:4d}\n\n"
        f"Classification report:\n{report}\n"
        f"Model saved -> {FUSION_DIR / 'best_fusion.pt'}\n"
    )

    print(result)
    with open(FUSION_DIR / "fusion_report.txt", "w") as f:
        f.write(result)
