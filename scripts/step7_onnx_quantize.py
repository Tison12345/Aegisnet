"""
Step 7: ONNX export + INT8 dynamic quantisation + latency benchmark.

Steps
-----
1. Export TCN, ST-GCN, and AttentionFusion to ONNX (FP32).
2. Apply INT8 dynamic post-training quantisation via ONNX Runtime.
3. Benchmark 100-pass mean latency for FP32 and INT8 on CPU.
4. Compare file sizes before and after quantisation.
5. Run INT8 fusion end-to-end on the test set and confirm accuracy
   is within 1 pp of the PyTorch baseline (99.33 %).

Notes
-----
- All models exported with batch_size=1 (real-time single-driver inference).
  The ST-GCN forward uses x.shape[0] in reshape ops, so batch is fixed in
  the ONNX graph — acceptable for single-driver real-time use.
- Dynamic INT8 quantises weight tensors offline; activations are quantised
  at runtime, so no calibration dataset is needed.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ── import model classes ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from step4_train_tcn import (DROPOUT, FEATURES as FACE_FEATURES,
                              KERNEL_SIZE, TCN, TCN_CHANNELS, WINDOW, STRIDE,
                              build_sequences)
from pose_stgcn_pipeline import (FEATURES_PER_LANDMARK, N_LANDMARKS,
                                  STGCN, make_adjacency_matrix, SEQ_LEN)
from step5_fusion import AttentionFusion, load_and_merge, POSE_COLS, LABEL_MAP, ROLL_CLIP

# ── paths ────────────────────────────────────────────────────────────────────────
ROOT         = Path(r"C:\Ageisnet")
DATA         = ROOT / "data"
MODELS       = DATA / "models"
ONNX_DIR     = MODELS / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

TCN_PT       = MODELS / "tcn"    / "best_tcn.pt"
TCN_SCALER   = MODELS / "tcn"    / "scaler.pkl"
STGCN_PT     = MODELS / "stgcn"  / "best_stgcn.pt"
FUSION_PT    = MODELS / "fusion" / "best_fusion.pt"

FACE_TRAIN   = DATA / "train.csv"
FACE_TEST    = DATA / "test.csv"
POSE_TRAIN   = DATA / "pose_train.csv"
POSE_TEST    = DATA / "pose_test.csv"

DEVICE       = torch.device("cpu")
BENCHMARK_N  = 100        # forward passes for latency measurement


# ── model loaders ────────────────────────────────────────────────────────────────
def load_tcn() -> TCN:
    m = TCN(len(FACE_FEATURES), TCN_CHANNELS, KERNEL_SIZE, DROPOUT)
    m.load_state_dict(torch.load(TCN_PT, weights_only=True, map_location=DEVICE))
    return m.eval()


def load_stgcn() -> STGCN:
    A = make_adjacency_matrix().to(DEVICE)
    m = STGCN(in_channels=FEATURES_PER_LANDMARK, num_classes=2, A=A)
    m.load_state_dict(torch.load(STGCN_PT, weights_only=True, map_location=DEVICE))
    return m.eval()


def load_fusion() -> AttentionFusion:
    m = AttentionFusion()
    m.load_state_dict(torch.load(FUSION_PT, weights_only=True, map_location=DEVICE))
    return m.eval()


# ── ONNX export ──────────────────────────────────────────────────────────────────
def export_tcn(model: TCN, out: Path):
    dummy = torch.randn(1, len(FACE_FEATURES), WINDOW)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["face_features"],
        output_names=["logit"],
        dynamic_axes={"face_features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def export_stgcn(model: STGCN, out: Path):
    # Input: (batch, T, C, V) = (1, 45, 4, 33)
    dummy = torch.randn(1, SEQ_LEN, FEATURES_PER_LANDMARK, N_LANDMARKS)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["pose_sequence"],
        output_names=["class_logits"],
        # batch kept fixed at 1 due to reshape ops using x.shape[0]
        opset_version=17,
        do_constant_folding=True,
    )


def export_fusion(model: AttentionFusion, out: Path):
    dummy = torch.randn(1, 2)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["stream_probs"],
        output_names=["safety_score", "attention_weights"],
        dynamic_axes={"stream_probs": {0: "batch"},
                      "safety_score": {0: "batch"},
                      "attention_weights": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


# ── INT8 quantisation ─────────────────────────────────────────────────────────────
def quantise(fp32_path: Path, int8_path: Path):
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


# ── latency benchmark ────────────────────────────────────────────────────────────
def benchmark(onnx_path: Path, dummy_input: np.ndarray,
              input_name: str, n: int = BENCHMARK_N) -> float:
    """
    Run `n` forward passes with ONNX Runtime and return mean latency in ms.
    First 5 passes are warm-up and excluded from timing.
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])
    feed = {input_name: dummy_input}

    # warm-up
    for _ in range(5):
        sess.run(None, feed)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


# ── file size helper ──────────────────────────────────────────────────────────────
def mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576


# ── INT8 end-to-end accuracy on fusion test set ───────────────────────────────────
def eval_int8_fusion_accuracy(
    tcn_int8:    Path,
    stgcn_int8:  Path,
    fusion_int8: Path,
    face_scaler: StandardScaler,
    pose_scaler: StandardScaler,
) -> float:
    import pandas as pd

    # Load merged test data (same logic as step5_fusion.py)
    test_merged = load_and_merge(FACE_TEST, POSE_TEST)
    test_merged = test_merged.sort_values("label").reset_index(drop=True)
    test_merged["_path"] = test_merged.index.astype(str)

    # Build sequences
    xf_list, xp_list, yl = [], [], []
    for label in ("alert", "drowsy"):
        sub = test_merged[test_merged["label"] == label]
        fv  = face_scaler.transform(sub[FACE_FEATURES].values.astype(np.float32))
        pv  = pose_scaler.transform(sub[POSE_COLS].values.astype(np.float32))
        n   = len(sub)
        for start in range(0, n - WINDOW + 1, STRIDE):
            xf_list.append(fv[start: start + WINDOW])
            xp_list.append(pv[start: start + WINDOW]
                           .reshape(WINDOW, N_LANDMARKS, FEATURES_PER_LANDMARK))
            yl.append(LABEL_MAP[label])

    if not xf_list:
        raise ValueError("No test sequences built from common frames.")

    Xf = np.stack(xf_list).astype(np.float32)   # (N, W, F)
    Xp = np.stack(xp_list).astype(np.float32)   # (N, W, V, C)
    y  = np.array(yl, dtype=np.int64)

    # TCN INT8: input (1, F, W)
    tcn_sess  = ort.InferenceSession(str(tcn_int8),
                                     providers=["CPUExecutionProvider"])
    stgcn_sess = ort.InferenceSession(str(stgcn_int8),
                                      providers=["CPUExecutionProvider"])
    fus_sess  = ort.InferenceSession(str(fusion_int8),
                                     providers=["CPUExecutionProvider"])

    face_probs, body_probs = [], []
    for i in range(len(Xf)):
        # TCN expects (1, F, W)
        x_tcn = Xf[i].T[np.newaxis]             # (1, F, W) = (1, 5, 45)
        logit = tcn_sess.run(None, {"face_features": x_tcn})[0]
        face_probs.append(float(1 / (1 + np.exp(-logit[0, 0]))))  # sigmoid

        # ST-GCN expects (1, T, C, V) — transposed from (W, V, C)
        x_stgcn = Xp[i].transpose(0, 2, 1)[np.newaxis]  # (1, T, C, V)
        logits  = stgcn_sess.run(None, {"pose_sequence": x_stgcn})[0]  # (1, 2)
        probs   = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        body_probs.append(float(probs[0, 1]))

    fus_input  = np.array(list(zip(face_probs, body_probs)), dtype=np.float32)
    safety_scores = fus_sess.run(None, {"stream_probs": fus_input})[0]  # (N,)
    preds = (safety_scores < 0.5).astype(int)
    return accuracy_score(y, preds)


# ── main ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASELINE_ACC = 0.9933    # PyTorch fusion result from step5

    # ── 1. Export to ONNX ───────────────────────────────────────────────────────
    print("=" * 62)
    print("STEP 1 — ONNX EXPORT (FP32)")
    print("=" * 62)

    paths = {
        "tcn":    ONNX_DIR / "best_tcn.onnx",
        "stgcn":  ONNX_DIR / "best_stgcn.onnx",
        "fusion": ONNX_DIR / "best_fusion.onnx",
    }
    paths_int8 = {
        "tcn":    ONNX_DIR / "best_tcn_int8.onnx",
        "stgcn":  ONNX_DIR / "best_stgcn_int8.onnx",
        "fusion": ONNX_DIR / "best_fusion_int8.onnx",
    }

    print("Exporting TCN ...", end=" ", flush=True)
    export_tcn(load_tcn(), paths["tcn"])
    onnx.checker.check_model(str(paths["tcn"]))
    print("OK")

    print("Exporting ST-GCN ...", end=" ", flush=True)
    export_stgcn(load_stgcn(), paths["stgcn"])
    onnx.checker.check_model(str(paths["stgcn"]))
    print("OK")

    print("Exporting Fusion ...", end=" ", flush=True)
    export_fusion(load_fusion(), paths["fusion"])
    onnx.checker.check_model(str(paths["fusion"]))
    print("OK")

    # ── 2. INT8 dynamic quantisation ────────────────────────────────────────────
    print()
    print("=" * 62)
    print("STEP 2 — INT8 DYNAMIC QUANTISATION")
    print("=" * 62)

    for name in ("tcn", "stgcn", "fusion"):
        print(f"Quantising {name} ...", end=" ", flush=True)
        quantise(paths[name], paths_int8[name])
        print("OK")

    # ── 3. Latency benchmark ─────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"STEP 3 — LATENCY BENCHMARK  ({BENCHMARK_N} passes, CPU)")
    print("=" * 62)

    dummy_tcn   = np.random.randn(1, len(FACE_FEATURES), WINDOW).astype(np.float32)
    dummy_stgcn = np.random.randn(1, SEQ_LEN, FEATURES_PER_LANDMARK,
                                  N_LANDMARKS).astype(np.float32)
    dummy_fus   = np.random.randn(1, 2).astype(np.float32)

    benchmarks = {
        "TCN   FP32": (paths["tcn"],      dummy_tcn,   "face_features"),
        "TCN   INT8": (paths_int8["tcn"], dummy_tcn,   "face_features"),
        "STGCN FP32": (paths["stgcn"],    dummy_stgcn, "pose_sequence"),
        "STGCN INT8": (paths_int8["stgcn"],dummy_stgcn,"pose_sequence"),
        "FUSIN FP32": (paths["fusion"],   dummy_fus,   "stream_probs"),
        "FUSIN INT8": (paths_int8["fusion"],dummy_fus, "stream_probs"),
    }

    latencies = {}
    for label, (path, dummy, inp_name) in benchmarks.items():
        ms = benchmark(path, dummy, inp_name)
        latencies[label] = ms
        print(f"  {label} : {ms:6.3f} ms")

    print()
    print("  Speedup from INT8:")
    for name in ("TCN", "STGCN", "FUSIN"):
        fp_key = f"TCN   FP32" if name == "TCN" else (f"STGCN FP32" if name == "STGCN" else "FUSIN FP32")
        i8_key = f"TCN   INT8" if name == "TCN" else (f"STGCN INT8" if name == "STGCN" else "FUSIN INT8")
        fp = latencies[fp_key]
        i8 = latencies[i8_key]
        print(f"    {name}: {fp:.3f} ms -> {i8:.3f} ms  ({fp/i8:.2f}x)")

    # ── 4. File size comparison ─────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("STEP 4 — FILE SIZE COMPARISON")
    print("=" * 62)
    print(f"  {'Model':<18} {'FP32 (MB)':>10} {'INT8 (MB)':>10} {'Reduction':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for name in ("tcn", "stgcn", "fusion"):
        fp_mb  = mb(paths[name])
        i8_mb  = mb(paths_int8[name])
        pct    = (1 - i8_mb / fp_mb) * 100
        label  = name.upper()
        print(f"  {label:<18} {fp_mb:>10.2f} {i8_mb:>10.2f} {pct:>9.1f}%")

    # ── 5. INT8 end-to-end accuracy ─────────────────────────────────────────────
    print()
    print("=" * 62)
    print("STEP 5 — INT8 END-TO-END ACCURACY (fusion test set)")
    print("=" * 62)

    with open(TCN_SCALER, "rb") as f:
        face_scaler = pickle.load(f)

    import pandas as pd
    train_merged = load_and_merge(FACE_TRAIN, POSE_TRAIN)
    pose_scaler  = StandardScaler()
    pose_scaler.fit(train_merged[POSE_COLS].values.astype(np.float32))

    int8_acc = eval_int8_fusion_accuracy(
        paths_int8["tcn"], paths_int8["stgcn"], paths_int8["fusion"],
        face_scaler, pose_scaler,
    )

    delta = abs(int8_acc - BASELINE_ACC)
    within = "YES" if delta <= 0.01 else "NO"

    print(f"  PyTorch FP32 baseline  : {BASELINE_ACC:.4f}  ({BASELINE_ACC*100:.2f} %)")
    print(f"  ONNX INT8 accuracy     : {int8_acc:.4f}  ({int8_acc*100:.2f} %)")
    print(f"  Accuracy drop          : {delta*100:.3f} pp")
    print(f"  Within 1 pp threshold  : {within}")

    # ── Final summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    total_fp32_ms = (latencies["TCN   FP32"] + latencies["STGCN FP32"] +
                     latencies["FUSIN FP32"])
    total_int8_ms = (latencies["TCN   INT8"] + latencies["STGCN INT8"] +
                     latencies["FUSIN INT8"])
    print(f"  Full pipeline FP32 latency : {total_fp32_ms:.2f} ms")
    print(f"  Full pipeline INT8 latency : {total_int8_ms:.2f} ms")
    print(f"  End-to-end speedup         : {total_fp32_ms/total_int8_ms:.2f}x")
    print(f"  INT8 fusion accuracy       : {int8_acc*100:.2f} %  (baseline {BASELINE_ACC*100:.2f} %)")
    print()
    print("ONNX models saved to:", ONNX_DIR)
    for name in ("tcn", "stgcn", "fusion"):
        print(f"  {paths[name].name:<28} {mb(paths[name]):.2f} MB")
        print(f"  {paths_int8[name].name:<28} {mb(paths_int8[name]):.2f} MB")
