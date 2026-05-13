import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import mediapipe as mp

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = DATA_DIR / "features.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
BEST_MODEL = ROOT_DIR / "models" / "best_tcn.pt"

CLASSES = ["alert", "drowsy"]
SEED = 42
SEQ_LEN = 45
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
POSE_IDX = [1, 152, 33, 263, 61, 291]
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
], dtype=np.float64)

LABEL_TO_INDEX = {"alert": 0, "drowsy": 1}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def make_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )


def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * (C + 1e-6))


def mouth_aspect_ratio(landmarks, w, h):
    top = (landmarks[MOUTH_TOP].x * w, landmarks[MOUTH_TOP].y * h)
    bottom = (landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h)
    left = (landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h)
    right = (landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h)
    return math.dist(top, bottom) / (math.dist(left, right) + 1e-6)


def head_pose(landmarks, w, h):
    image_points = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_IDX],
        dtype=np.float64,
    )
    focal_length = float(w)
    camera_matrix = np.array(
        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = 0.0
    return pitch, yaw, roll


def process_image(face_mesh, img_path: Path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    avg_ear = (
        eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        + eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ) / 2.0
    mouth_ar = mouth_aspect_ratio(landmarks, w, h)
    pitch, yaw, roll = head_pose(landmarks, w, h)
    if pitch is None:
        return None
    return avg_ear, mouth_ar, pitch, yaw, roll


def extract_features():
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    face_mesh = make_face_mesh()

    for label in CLASSES:
        folder = PROCESSED_DIR / label
        images = sorted(folder.glob("*.png"))
        print(f"Processing {label}: {len(images)} files")
        skipped = 0
        for idx, img_path in enumerate(images, start=1):
            result = process_image(face_mesh, img_path)
            if result is None:
                skipped += 1
                continue
            ear_val, mar_val, pitch, yaw, roll = result
            rows.append({
                "image_path": str(img_path),
                "label": label,
                "EAR": round(ear_val, 4),
                "MAR": round(mar_val, 4),
                "pitch": round(pitch, 4),
                "yaw": round(yaw, 4),
                "roll": round(roll, 4),
            })
            if idx % 500 == 0:
                print(f"  {idx}/{len(images)} processed, skipped={skipped}")
        print(f"  {label} done: skipped {skipped}/{len(images)}")

    face_mesh.close()
    with open(FEATURES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label", "EAR", "MAR", "pitch", "yaw", "roll"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} feature rows to {FEATURES_CSV}")


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
    print(f"Train rows: {len(train)} saved to {TRAIN_CSV}")
    print(f"Test rows: {len(test)} saved to {TEST_CSV}")
    print("Train label distribution:\n", train["label"].value_counts())
    print("Test label distribution:\n", test["label"].value_counts())


class SequenceDataset(Dataset):
    def __init__(self, seq_len: int, scaler=None, csv_path: Optional[Path] = None, data_frame: Optional[pd.DataFrame] = None):
        self.seq_len = seq_len
        self.scaler = scaler
        if data_frame is not None:
            self.df = data_frame
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("csv_path or data_frame is required to build sequences")
        self.sequences, self.labels = self._build_sequences()
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences of length {seq_len} could be built from the provided data")

    def _build_sequences(self):
        sequences = []
        labels = []
        for label, group in self.df.groupby("label"):
            group = group.sort_values("image_path")
            features = group[["EAR", "MAR", "pitch", "yaw", "roll"]].to_numpy(dtype=np.float32)
            for start in range(0, len(features) - self.seq_len + 1):
                seq = features[start : start + self.seq_len]
                sequences.append(seq)
                labels.append(LABEL_TO_INDEX[label])
        sequences = np.stack(sequences) if sequences else np.zeros((0, self.seq_len, 5), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        if self.scaler is not None and len(sequences) > 0:
            flat = sequences.reshape(-1, sequences.shape[-1])
            scaled = self.scaler.transform(flat).reshape(sequences.shape)
            sequences = scaled
        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class TemporalConvNet(nn.Module):
    def __init__(self, input_size: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.mean(dim=2)
        return self.classifier(x)


def train_tcn():
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loading {len(train_df)} training feature rows")

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=SEED,
        stratify=train_df["label"],
    )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_features = train_df[["EAR", "MAR", "pitch", "yaw", "roll"]].to_numpy(dtype=np.float32)
    scaler.fit(train_features)

    train_dataset = SequenceDataset(seq_len=SEQ_LEN, scaler=scaler, data_frame=train_df)
    val_dataset = SequenceDataset(seq_len=SEQ_LEN, scaler=scaler, data_frame=val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TemporalConvNet(input_size=5, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
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
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_batch.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        print(
            f"Epoch {epoch:02d}/{EPOCHS}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  New best model saved to {BEST_MODEL} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


def evaluate_model():
    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Best model not found at {BEST_MODEL}")
    model = TemporalConvNet(input_size=5, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_df = pd.read_csv(TEST_CSV)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_df = pd.read_csv(TRAIN_CSV)
    scaler.fit(train_df[["EAR", "MAR", "pitch", "yaw", "roll"]].to_numpy(dtype=np.float32))

    test_dataset = SequenceDataset(seq_len=SEQ_LEN, scaler=scaler, csv_path=TEST_CSV)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
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
    parser = argparse.ArgumentParser(description="Drowsiness TCN pipeline")
    parser.add_argument("--extract", action="store_true", help="Compute features.csv from processed frames")
    parser.add_argument("--split", action="store_true", help="Split features.csv into train.csv and test.csv")
    parser.add_argument("--train", action="store_true", help="Train the TCN model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best_tcn.pt on test.csv")
    parser.add_argument("--all", action="store_true", help="Run all steps: extract, split, train, evaluate")
    args = parser.parse_args()

    if args.all or args.extract:
        extract_features()
    if args.all or args.split:
        split_dataset()
    if args.all or args.train:
        train_tcn()
    if args.all or args.evaluate:
        evaluate_model()

    if not any([args.all, args.extract, args.split, args.train, args.evaluate]):
        parser.print_help()
