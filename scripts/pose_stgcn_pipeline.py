import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = DATA_DIR / "pose_features.csv"
TRAIN_CSV = DATA_DIR / "pose_train.csv"
TEST_CSV = DATA_DIR / "pose_test.csv"
BEST_MODEL = ROOT_DIR / "data" / "models" / "stgcn" / "best_stgcn.pt"

CLASSES = ["alert", "drowsy"]
SEED = 42
SEQ_LEN = 45
BATCH_SIZE = 32
EPOCHS = 30
STRIDE = 15       # sliding-window stride; stride=1 creates ~33k seqs and exhausts RAM on CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LANDMARKS = 33
FEATURES_PER_LANDMARK = 4

LABEL_TO_INDEX = {"alert": 0, "drowsy": 1}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (3, 7), (6, 8),
    (0, 9), (0, 10), (9, 10),
    (0, 11), (0, 12), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

EDGE_LIST = POSE_EDGES + [(j, i) for i, j in POSE_EDGES]


def make_pose_detector():
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )


def process_image(pose, img_path: Path):
    import cv2

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None

    row = []
    for landmark in results.pose_landmarks.landmark:
        row.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility,
        ])
    return row


def extract_pose_features():
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pose = make_pose_detector()

    for label in CLASSES:
        folder = PROCESSED_DIR / label
        images = sorted(folder.glob("*.png"))
        print(f"Processing {label}: {len(images)} frames")
        skipped = 0
        for idx, img_path in enumerate(images, start=1):
            row = process_image(pose, img_path)
            if row is None:
                skipped += 1
                continue
            record = {
                "image_path": str(img_path),
                "label": label,
            }
            for i, val in enumerate(row):
                record[f"v{i}"] = round(float(val), 6)
            rows.append(record)
            if idx % 500 == 0:
                print(f"  {idx}/{len(images)} processed, skipped={skipped}")

        print(f"  {label} done. skipped {skipped}/{len(images)}")

    pose.close()
    fieldnames = ["image_path", "label"] + [f"v{i}" for i in range(N_LANDMARKS * FEATURES_PER_LANDMARK)]
    with open(FEATURES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {FEATURES_CSV}")


def split_dataset():
    df = pd.read_csv(FEATURES_CSV)
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"Saved {len(train)} train rows to {TRAIN_CSV}")
    print(f"Saved {len(test)} test rows to {TEST_CSV}")
    print("Train distribution:\n", train["label"].value_counts())
    print("Test distribution:\n", test["label"].value_counts())


class PoseSequenceDataset(Dataset):
    def __init__(self, seq_len: int, scaler: Optional[StandardScaler] = None, csv_path: Optional[Path] = None, data_frame: Optional[pd.DataFrame] = None):
        self.seq_len = seq_len
        self.scaler = scaler
        if data_frame is not None:
            self.df = data_frame
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("csv_path or data_frame is required")
        self.sequences, self.labels = self._build_sequences()
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences of length {seq_len} could be built")

    def _build_sequences(self):
        sequences = []
        labels = []
        feature_cols = [f"v{i}" for i in range(N_LANDMARKS * FEATURES_PER_LANDMARK)]
        for label, group in self.df.groupby("label"):
            group = group.sort_values("image_path")
            features = group[feature_cols].to_numpy(dtype=np.float32)
            if self.scaler is not None and features.shape[0] > 0:
                features = self.scaler.transform(features)
            for start in range(0, len(features) - self.seq_len + 1, STRIDE):
                seq = features[start : start + self.seq_len]
                seq = seq.reshape(self.seq_len, N_LANDMARKS, FEATURES_PER_LANDMARK)
                sequences.append(seq)
                labels.append(LABEL_TO_INDEX[label])
        if sequences:
            sequences = np.stack(sequences)
        else:
            sequences = np.zeros((0, self.seq_len, N_LANDMARKS, FEATURES_PER_LANDMARK), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        seq = seq.permute(0, 2, 1)
        return seq, torch.tensor(self.labels[idx], dtype=torch.long)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, kernel_size: int = 9, dropout: float = 0.3):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return nn.functional.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, A: torch.Tensor):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * N_LANDMARKS)
        self.layer1 = STGCNBlock(in_channels, 32, A)
        self.layer2 = STGCNBlock(32, 64, A)
        self.layer3 = STGCNBlock(64, 128, A)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        N, T, C, V = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=2).mean(dim=2)
        return self.fc(x)


def make_adjacency_matrix():
    A = np.zeros((N_LANDMARKS, N_LANDMARKS), dtype=np.float32)
    for i, j in EDGE_LIST:
        A[i, j] = 1.0
    A = A + np.eye(N_LANDMARKS, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv = np.diag(1.0 / np.sqrt(np.where(D > 0, D, 1.0)))
    A_norm = D_inv @ A @ D_inv
    return torch.tensor(A_norm, dtype=torch.float32)


def train_stgcn():
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=SEED,
        stratify=df["label"],
    )

    scaler = StandardScaler()
    train_features = train_df[[f"v{i}" for i in range(N_LANDMARKS * FEATURES_PER_LANDMARK)]].to_numpy(dtype=np.float32)
    scaler.fit(train_features)

    train_dataset = PoseSequenceDataset(seq_len=SEQ_LEN, scaler=scaler, data_frame=train_df)
    val_dataset = PoseSequenceDataset(seq_len=SEQ_LEN, scaler=scaler, data_frame=val_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    A = make_adjacency_matrix().to(DEVICE)
    model = STGCN(in_channels=FEATURES_PER_LANDMARK, num_classes=2, A=A).to(DEVICE)

    # Weight drowsy class by alert/drowsy ratio to handle class imbalance
    n_alert  = (train_df["label"] == "alert").sum()
    n_drowsy = (train_df["label"] == "drowsy").sum()
    weight   = torch.tensor([1.0, n_alert / n_drowsy], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_ctr = 0
    PATIENCE = 7

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ST-GCN params={total_params:,}  device={DEVICE}")
    print(f"Train sequences={len(train_dataset)}  Val sequences={len(val_dataset)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                logits = model(x_batch)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(y_batch.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  -> Best saved (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


def evaluate_stgcn():
    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Best model not found at {BEST_MODEL}")
    scaler = StandardScaler()
    train_df = pd.read_csv(TRAIN_CSV)
    scaler.fit(train_df[[f"v{i}" for i in range(N_LANDMARKS * FEATURES_PER_LANDMARK)]].to_numpy(dtype=np.float32))

    test_dataset = PoseSequenceDataset(seq_len=SEQ_LEN, scaler=scaler, csv_path=TEST_CSV)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    A = make_adjacency_matrix().to(DEVICE)
    model = STGCN(in_channels=FEATURES_PER_LANDMARK, num_classes=2, A=A).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE, weights_only=True))
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y_batch.numpy())

    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="binary")
    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, target_names=["alert", "drowsy"])
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlazePose ST-GCN drowsiness pipeline")
    parser.add_argument("--extract", action="store_true", help="Extract pose_features.csv from processed frames")
    parser.add_argument("--split", action="store_true", help="Split pose_features.csv into pose_train.csv and pose_test.csv")
    parser.add_argument("--train", action="store_true", help="Train the ST-GCN model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best_stgcn.pt on pose_test.csv")
    parser.add_argument("--all", action="store_true", help="Run all steps: extract, split, train, evaluate")
    args = parser.parse_args()

    if args.all or args.extract:
        extract_pose_features()
    if args.all or args.split:
        split_dataset()
    if args.all or args.train:
        train_stgcn()
    if args.all or args.evaluate:
        evaluate_stgcn()

    if not any([args.all, args.extract, args.split, args.train, args.evaluate]):
        parser.print_help()
