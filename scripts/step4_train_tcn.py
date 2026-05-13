"""
Step 4: Train a Temporal Convolutional Network (TCN) on face-stream features.

Input:  data/train.csv, data/test.csv
        Features: EAR, MAR, pitch, yaw, roll  (5 channels)
        Label:    alert=0, drowsy=1

Approach:
  - Sort frames by path (preserves temporal order within each class)
  - Build overlapping sliding windows of W frames with stride S
  - Normalise with StandardScaler fit on train only
  - TCN: 4 dilated causal residual blocks → global-avg-pool → binary output
  - Outputs: saved model, scaler, per-epoch loss/accuracy curves, test report

Output files (all under data/models/tcn/):
  best_tcn.pt       -- best model weights (by val loss)
  scaler.pkl        -- fitted StandardScaler
  history.csv       -- per-epoch train/val loss and accuracy
  test_report.txt   -- classification report + confusion matrix on test set
"""

import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── config ──────────────────────────────────────────────────────────────────────
TRAIN_CSV  = Path(r"C:\Ageisnet\data\train.csv")
TEST_CSV   = Path(r"C:\Ageisnet\data\test.csv")
OUT_DIR    = Path(r"C:\Ageisnet\data\models\tcn")

FEATURES   = ["EAR", "MAR", "pitch", "yaw", "roll"]
WINDOW     = 45        # frames per sequence (~1.5 s @ 30 fps / 3 s @ 15 fps)
STRIDE     = 15        # sliding window stride (67% overlap)
ROLL_CLIP  = 90.0      # clip roll outliers before scaling

BATCH_SIZE = 128
LR         = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS     = 40
VAL_FRAC   = 0.1       # fraction of train sequences used for validation
PATIENCE   = 7         # early-stop patience (epochs without val loss improvement)
SEED       = 42

# TCN hyper-params
TCN_CHANNELS  = [64, 64, 128, 128]   # channels per residual block
KERNEL_SIZE   = 3
DROPOUT       = 0.2

DEVICE = torch.device("cpu")          # CPU-only laptop


# ── dataset ─────────────────────────────────────────────────────────────────────
def build_sequences(df: pd.DataFrame, window: int, stride: int):
    """
    Return (X, y) where X has shape (N, window, n_features).
    Sequences are built within each class independently so no label boundary
    is crossed.  Frames are sorted by path to preserve temporal order.
    """
    X_list, y_list = [], []
    for label, numeric in (("alert", 0), ("drowsy", 1)):
        sub = (df[df["label"] == label]
               .sort_values("frame_path")
               .reset_index(drop=True))
        vals = sub[FEATURES].values.astype(np.float32)
        n = len(vals)
        for start in range(0, n - window + 1, stride):
            X_list.append(vals[start : start + window])
            y_list.append(numeric)
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, W, F)  →  model expects (N, F, W) for Conv1d
        self.X = torch.from_numpy(X).permute(0, 2, 1)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── model ────────────────────────────────────────────────────────────────────────
class CausalBlock(nn.Module):
    """
    One dilated causal residual block:
      input → pad → Conv1d(dilation) → BN → ReLU → dropout
            → pad → Conv1d(dilation) → BN → ReLU → dropout
      + residual (1×1 conv if channels differ)
    """
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation   # causal: pad only on the left

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                               padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()

        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else nn.Identity())
        self.pad = pad

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)[:, :, : -self.pad or None]))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)[:, :, : -self.pad or None]))
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
        self.net    = nn.Sequential(*layers)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Linear(channels[-1], 1)

    def forward(self, x):
        # x: (B, F, W)
        out = self.net(x)           # (B, C_last, W)
        out = self.pool(out)        # (B, C_last, 1)
        out = out.squeeze(-1)       # (B, C_last)
        return self.head(out)       # (B, 1) — raw logit


# ── helpers ──────────────────────────────────────────────────────────────────────
def accuracy(logits, labels):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        total_acc  += accuracy(out, y)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out  = model(X)
        total_loss += criterion(out, y).item()
        total_acc  += accuracy(out, y)
    n = len(loader)
    return total_loss / n, total_acc / n


# ── main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & preprocess -------------------------------------------------------
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # Clip roll outliers (gimbal-lock artefacts)
    for df in (train_df, test_df):
        df["roll"] = df["roll"].clip(-ROLL_CLIP, ROLL_CLIP)

    # 2. Build sequences ---------------------------------------------------------
    print(f"Building sequences  window={WINDOW}  stride={STRIDE} ...")
    X_train_all, y_train_all = build_sequences(train_df, WINDOW, STRIDE)
    X_test,  y_test          = build_sequences(test_df,  WINDOW, STRIDE)
    print(f"  train sequences: {len(X_train_all)}"
          f"  (alert={int((y_train_all==0).sum())}, drowsy={int((y_train_all==1).sum())})")
    print(f"  test  sequences: {len(X_test)}")

    # 3. Normalise (fit on train only) -------------------------------------------
    N, W, F = X_train_all.shape
    scaler  = StandardScaler()
    X_flat  = X_train_all.reshape(-1, F)
    scaler.fit(X_flat)

    X_train_all = scaler.transform(X_flat).reshape(N, W, F).astype(np.float32)
    Nt, Wt, Ft  = X_test.shape
    X_test      = scaler.transform(X_test.reshape(-1, Ft)).reshape(Nt, Wt, Ft).astype(np.float32)

    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  Scaler saved.")

    # 4. Train / val split -------------------------------------------------------
    rng     = np.random.default_rng(SEED)
    idx     = rng.permutation(len(X_train_all))
    n_val   = max(1, int(len(idx) * VAL_FRAC))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    train_ds = SequenceDataset(X_train_all[tr_idx], y_train_all[tr_idx])
    val_ds   = SequenceDataset(X_train_all[val_idx], y_train_all[val_idx])
    test_ds  = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5. Model, optimiser, loss --------------------------------------------------
    model     = TCN(len(FEATURES), TCN_CHANNELS, KERNEL_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # Class imbalance: weight drowsy class slightly higher
    n_alert  = int((y_train_all[tr_idx] == 0).sum())
    n_drowsy = int((y_train_all[tr_idx] == 1).sum())
    pos_weight = torch.tensor([n_alert / n_drowsy], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTCN  params={total_params:,}  device={DEVICE}")
    print(f"Training {len(train_ds)} sequences, validating on {len(val_ds)} ...")

    # 6. Training loop -----------------------------------------------------------
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5), "train_acc": round(tr_acc, 4),
            "val_loss":   round(vl_loss, 5), "val_acc":   round(vl_acc, 4),
        })
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), OUT_DIR / "best_tcn.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    # 7. Save training history ---------------------------------------------------
    with open(OUT_DIR / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # 8. Evaluate on test set ----------------------------------------------------
    model.load_state_dict(torch.load(OUT_DIR / "best_tcn.pt", weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            preds  = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().flatten().astype(int).tolist())

    report = classification_report(all_labels, all_preds,
                                   target_names=["alert", "drowsy"])
    cm     = confusion_matrix(all_labels, all_preds)

    report_txt = (
        "=== TCN Test Results ===\n\n"
        f"Window={WINDOW}  Stride={STRIDE}  Channels={TCN_CHANNELS}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=actual, cols=predicted):\n"
        f"             alert   drowsy\n"
        f"  alert     {cm[0,0]:6d}   {cm[0,1]:6d}\n"
        f"  drowsy    {cm[1,0]:6d}   {cm[1,1]:6d}\n"
    )
    print("\n" + report_txt)

    with open(OUT_DIR / "test_report.txt", "w") as f:
        f.write(report_txt)

    print(f"\nAll outputs saved to {OUT_DIR}")
