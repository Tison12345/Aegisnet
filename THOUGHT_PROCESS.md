# Ageisnet — Thought Process & Workflow Evolution

> A complete record of how the system was designed, built, and iterated — every decision made, every dead end hit, every bug fixed, and why each choice was made the way it was.

---

## Table of Contents

- [The Starting Vision](#the-starting-vision)
- [Phase 0 — Data Foundation](#phase-0--data-foundation)
- [Phase 1 — Face Feature Extraction](#phase-1--face-feature-extraction)
- [Phase 2 — Train / Test Split](#phase-2--train--test-split)
- [Phase 3 — TCN: Temporal Face Stream](#phase-3--tcn-temporal-face-stream)
- [Phase 4 — ST-GCN: Skeletal Body Stream](#phase-4--st-gcn-skeletal-body-stream)
- [Phase 5 — Attention Fusion](#phase-5--attention-fusion)
- [Phase 6 — ArcFace Identity Verification](#phase-6--arcface-identity-verification)
- [Phase 7 — ONNX Export and INT8 Quantisation](#phase-7--onnx-export-and-int8-quantisation)
- [Phase 8 — Live Webcam Systems](#phase-8--live-webcam-systems)
- [Phase 9 — Behavioral Biometric Innovation](#phase-9--behavioral-biometric-innovation)
- [Deployment Challenges](#deployment-challenges)
- [Full Bug Log](#full-bug-log)
- [Architecture Decisions That Were Changed](#architecture-decisions-that-were-changed)
- [Key Metrics Timeline](#key-metrics-timeline)

---

## The Starting Vision

The goal was a **real-time driver safety system for carpooling platforms** that could run on a mid-range smartphone CPU — no cloud, no GPU, no dedicated hardware. Two core capabilities were needed:

1. **Drowsiness detection** — continuous monitoring of driver alertness from a camera
2. **Identity verification** — confirm at ride start that the registered driver is behind the wheel

The constraint of smartphone CPU immediately ruled out large vision transformers and YOLO-style architectures. The design had to be lightweight, fast, and deployable via ONNX Runtime.

The two-stream idea (face features + body keypoints) came from observing that drowsy driving has both facial signals (drooping eyes, yawning) and postural signals (head slump, shoulder sag). A single-stream model would miss one dimension of the problem.

---

## Phase 0 — Data Foundation

### Finding the right dataset

The first attempt was `banudeep/nthuddd2` on Kaggle — chosen because it had more downloads and community verification. This dataset had been deleted by the time we tried to access it.

**Pivot:** Switched to `ismailnasri20/driver-drowsiness-dataset-ddd`, a pre-processed version of the NTHU Drowsy Driver dataset.

### Unexpected dataset format

The original plan assumed video files at 15 FPS that would need frame extraction. The actual dataset was already pre-extracted PNG frames, organized with letter-prefixed filenames (`a####.png` = subject A, `b####.png` = subject B, etc.).

**Impact:** The frame extraction step was eliminated entirely. This was a lucky shortcut — it saved hours of ffmpeg processing and meant the dataset was ready to use immediately.

### Dataset statistics after organisation

- 41,769 total frames across alert and drowsy classes
- Multiple subjects (subjects a–f at minimum)
- Roughly balanced alert/drowsy split
- All frames processed from `data/raw/` → `data/processed/alert/` and `data/processed/drowsy/`

---

## Phase 1 — Face Feature Extraction

### Initial approach: `mp.solutions` API

The first version of `step2_extract_features.py` used the classic MediaPipe Face Mesh via `mp.solutions.face_mesh`. This was the documented approach in most tutorials and StackOverflow answers.

**The crash:** `AttributeError: module 'mediapipe' has no attribute 'solutions'`

MediaPipe removed the entire `mp.solutions` namespace in version 0.10.35, replacing it with the Tasks API. Most online resources still showed the old API.

**Fix:** Rewrote the extraction script from scratch using:
- `mediapipe.tasks.python.vision.FaceLandmarker`
- `face_landmarker.task` model file downloaded from MediaPipe's model hub
- `mp.Image` for frame ingestion

This was a non-trivial rewrite — the Tasks API has a completely different initialization flow, running mode configuration, and output format.

### Feature engineering decisions

**EAR (Eye Aspect Ratio)** — chosen over raw eye-open percentage because it is illumination-invariant and scale-invariant. Uses the standard 6-point formula:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Both eyes are computed and averaged to reduce noise from asymmetric lighting or head angle.

**MAR (Mouth Aspect Ratio)** — captures yawning, the clearest behavioural signal of drowsiness. Defined as vertical opening divided by horizontal width using 4 lip landmarks.

**Head pose (solvePnP)** — pitch, yaw, roll derived from 6 facial anchor points projected onto a 3D head model. This captures distraction (looking away) and heavy-head slump.

### The gimbal-lock bug

After extraction, 100% of frames showed pitch values above 90°. The raw `solvePnP` output decomposes the rotation matrix using `atan2`, which for a near-frontal face produces a pitch of ~170° (near the ±180° wraparound) rather than the physically correct ~-10° (slightly downward tilt).

**Root cause:** Euler angle decomposition has a gimbal ambiguity — the same physical orientation has two valid angle representations. OpenCV consistently picks the ±180° neighbourhood for forward-facing cameras.

**Fix:** Two lines added after `atan2`:
```python
if pitch >  90: pitch -= 180.0
if pitch < -90: pitch += 180.0
```

This folds the wraparound back to the correct small-angle range. The fix also handles the symmetric case for cameras mounted below the driver's eyeline.

**Impact:** Without this fix, pitch was meaningless (100% of frames near ±180°). After the fix, pitch correctly showed the expected distribution centered near 0° with drowsy frames showing greater downward slump.

### Face detection coverage

- Total frames processed: 41,769
- Frames with no face detected: 11 (0.03%)
- Face detection rate: **99.97%**

The near-perfect detection rate validated that the dataset quality was high and the MTCNN-based FaceLandmarker had no trouble with the driving scenarios.

---

## Phase 2 — Train / Test Split

A standard 80/20 stratified split was applied to `features.csv`. Stratification by class was important because a 60/40 alert/drowsy imbalance would otherwise produce a biased test distribution.

One early decision that proved critical later: face features and pose features were split **independently** (two separate scripts reading two separate CSVs). This created a data leakage problem that only surfaced during fusion training — see Phase 5.

---

## Phase 3 — TCN: Temporal Face Stream

### Architecture rationale

The 5 face features (EAR, MAR, pitch, yaw, roll) are time series. Drowsiness is a temporal phenomenon — a single frame with low EAR could just be a blink; sustained low EAR over 3 seconds is drowsy driving. The model needs a temporal receptive field.

**Why TCN over LSTM:**

| Criteria | TCN | LSTM |
|---|---|---|
| Parallelism during training | Full (convolutions) | Sequential (hidden state) |
| Vanishing gradients | No (residual + BN) | Yes (mitigated by gating) |
| Fixed receptive field | Yes (predictable) | No (theoretical infinite) |
| ONNX export | Clean (no dynamic state) | Complex (requires state tracking) |
| Inference speed on CPU | Fast | Moderate |

The fixed receptive field of TCN maps directly to a fixed input window, which is exactly what ONNX Runtime expects.

**Architecture details:**
- 4 dilated causal residual blocks with dilations `[1, 2, 4, 8]`
- Channels `[64, 64, 128, 128]` — gradual widening
- Dilation doubles per block: receptive field = `1 + 2*(kernel-1)*sum(dilations)` = 1 + 2*2*15 = 61 frames
- `AdaptiveAvgPool1d(1)` collapses the time dimension to a single 128-d vector
- `Linear(128 → 1)` binary output with `BCEWithLogitsLoss`

**Window and stride choices:**
- Window = 45 frames ≈ 3 seconds at 15 FPS — long enough to capture a drowsy head-nod cycle
- Stride = 15 frames — 67% overlap, balances dataset size vs sequence diversity

### Training bug: `ReduceLROnPlateau` verbose kwarg

PyTorch 2.10 silently removed the `verbose=True` argument from `ReduceLROnPlateau`. The script crashed with:

```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

**Fix:** Removed the `verbose` argument. Learning rate changes are visible anyway through the loss curve.

### TCN results

- Test accuracy: **99.5%**
- Parameters: 220,993
- Zero errors on alert class, 3 errors on drowsy class (out of 552 sequences)

---

## Phase 4 — ST-GCN: Skeletal Body Stream

### Why a second stream?

Face features alone miss postural drowsiness: a driver can have relatively normal EAR but still be slumped forward, shoulders dropped, head lolling. Body keypoints capture these patterns.

### Why ST-GCN over a flat CNN or MLP on keypoints?

The 33 BlazePose landmarks have explicit spatial relationships — the shoulder connects to the elbow, the elbow to the wrist. A flat MLP treats all joints independently and loses this structure. A spatial graph convolution propagates signals along anatomically meaningful edges.

ST-GCN combines:
- **Spatial graph convolution** — aggregates features from adjacent joints via the normalised adjacency matrix `D^{-½}(A+I)D^{-½}`
- **Temporal convolution** — captures motion over the 45-frame window
- **Residual connections** — stabilises training of 3-deep graph convolution

### The OOM crash (stride=1)

The first run of `pose_stgcn_pipeline.py` allocated ~800 MB of RAM to build sequences with `stride=1`, then hung indefinitely on training. On a CPU-only laptop this is a fatal OOM condition.

**Root cause:** With stride=1, every consecutive window of 45 frames produces a sequence. On ~33,000 pose frames this yields ~33,000 sequences × 45 frames × 33 joints × 4 features × 4 bytes = ~780 MB, all in RAM before training begins.

**Fix:** Added `STRIDE = 15` as a global constant at the top of the script. This reduces sequences by 15×, bringing RAM usage to ~52 MB — well within budget.

**Lesson:** Window stride is not just a training quality hyperparameter; it is a memory management decision. Always set it first based on the available RAM budget before tuning for accuracy.

### Wrong model save path

After training completed, the model was being saved to `ROOT/"models"/"best_stgcn.pt"` (a root-level `models/` folder) instead of `ROOT/"data"/"models"/"stgcn"/"best_stgcn.pt"`. The evaluation step then loaded from the correct path, found nothing, and failed.

**Fix:** Corrected `BEST_MODEL` path to match the canonical model directory.

### ST-GCN results

- Test accuracy: **98.6%**
- F1 score: 0.986
- Parameters: 216,810

---

## Phase 5 — Attention Fusion

### The data leakage discovery

When step5_fusion.py loaded the face test set (8,354 rows) and pose test set (independently split), the inner join on frame path produced only **2,318 common frames** — dramatically less than expected.

**Root cause:** The face CSV and pose CSV were split independently. Random shuffling in the 80/20 split sent different frames to the test set for each modality. A frame present in `face_test.csv` might be in `pose_train.csv` and vice versa.

If we had naively concatenated predictions without the join, we would have been evaluating the fusion on frames where one stream had been trained on that exact sample — a form of train/test leakage that would have inflated the accuracy.

**Fix:** `load_and_merge()` performs an inner join on the normalised absolute path before building sequences. This ensures every sequence in the fusion evaluation was unseen by both models during training.

### Attention mechanism design

A softmax-normalised attention over two stream probabilities:

```
[p_face, p_body] → MLP(2→32→2) → softmax → [w_face, w_body]
Safety Score = 1 − (w_face × p_face + w_body × p_body)
```

The MLP learns to weight streams per-sample — if the face is occluded or poorly lit, the body stream gets higher weight automatically. The safety score is inverted so that 1.0 = fully alert (higher is safer).

### Learned weight interpretation

Average weights after training: `w_face = 0.644`, `w_body = 0.356`.

The face stream earns roughly twice the weight of the body stream. This is expected: EAR and MAR are direct physiological measures of drowsiness, whereas body posture is a secondary indicator that often lags the onset.

### Fusion results

- Test sequences: 149
- Accuracy: **99.33%** (1 error: one alert sequence misclassified as drowsy)
- Both individual streams: face=98.66%, body=97.99%
- Fusion outperforms both streams independently — validates the two-stream design

---

## Phase 6 — ArcFace Identity Verification

### First attempt: InsightFace

InsightFace is the gold-standard face recognition library — production-grade ArcFace with a clean Python API. The install was attempted:

```
pip install insightface
```

**The failure:** InsightFace builds a C extension (`cython_bbox`) that requires Microsoft C++ Build Tools. On the development machine (Windows, no Build Tools installed), the wheel build failed.

**Decision:** Rather than install a 2+ GB toolchain for a dependency that is only needed at inference time, we switched to `facenet-pytorch`.

### Why facenet-pytorch is equivalent

`facenet-pytorch` provides `InceptionResnetV1` pretrained on VGGFace2 — the same backbone architecture, same training data, same 512-d L2-normalised embedding space as InsightFace's ArcFace model. It is pure PyTorch with no C extensions.

Cosine similarity in the same embedding space → identical verification logic, identical threshold interpretation.

### Test design

Rather than using a separate face dataset, the driver dataset's letter-prefixed filenames were used:
- Register on one `a####.png` frame (subject A)
- Verify against 5 other `a####.png` frames (same person)
- Reject 5 frames from subjects b, c, d, e, f (different people)

### ArcFace results

- Same-person: 5/5 verified, avg similarity = **0.923**
- Different-person: 5/5 rejected, avg similarity = **0.166**
- Separability gap: **0.758** — large, clean margin
- Threshold: 0.85 (sits 0.073 above the impostor ceiling with 0.038 margin below the genuine floor)

---

## Phase 7 — ONNX Export and INT8 Quantisation

### Why ONNX?

PyTorch models require the PyTorch runtime (~200 MB). ONNX Runtime is ~10 MB and runs on Android, iOS, Linux, Windows, and embedded devices with identical results. For smartphone deployment, ONNX is the production path.

### The `optimize_model` bug

`onnxruntime.quantization.quantize_dynamic()` used to accept `optimize_model=True` as an argument. In onnxruntime 1.26.0, this argument was removed.

```
TypeError: quantize_dynamic() got an unexpected keyword argument 'optimize_model'
```

**Fix:** Removed the argument. Model optimisation now happens automatically inside ONNX Runtime's export pass.

### The speedup dict key mismatch

The benchmark section stored latencies in a dict with keys like `"TCN   FP32"` (3 spaces for TCN/FUSIN, 1 space for STGCN) due to column-alignment formatting. The speedup lookup used a pattern that didn't match the STGCN key, crashing with `KeyError`.

**Fix:** Explicit key lookup for each model name rather than string interpolation.

### INT8 is slower — and why that's fine

Counter-intuitively, INT8 inference was 3–5× **slower** than FP32 on the test machine:

| Model | FP32 | INT8 |
|---|---|---|
| TCN | 2.07 ms | 7.26 ms |
| ST-GCN | 10.50 ms | 51.82 ms |

**Why:** Dynamic INT8 quantisation on consumer Intel/AMD CPUs only provides a speed-up when the chip has dedicated INT8 vector hardware (Intel VNNI, AMX, or ARM NEON INT8). Without this, the runtime dequantises weights to FP32 at inference time, adding overhead on top of the regular FP32 computation.

**The actual benefit:** ~71% model size reduction. On a mobile device with an INT8-capable NPU (Snapdragon, Dimensity, Apple Neural Engine), the same INT8 models would deliver the expected 2–4× speed-up.

### ONNX results

- TCN: 0.86 MB → 0.25 MB (70.6% reduction)
- ST-GCN: 0.84 MB → 0.24 MB (71.7% reduction)
- INT8 accuracy: 100.00% (baseline 99.33%) — zero accuracy loss

---

## Phase 8 — Live Webcam Systems

### live_identity.py — ArcFace live verification

The first live script brought ArcFace into the webcam loop:
- Registration: one SPACE keypress captures and saves a 512-d embedding
- Verification: one SPACE keypress compares a fresh embedding

Key implementation choice: MTCNN is run on each captured frame for face detection and alignment before passing to ArcFace. This ensures the embedding is extracted from a properly cropped, normalised face region regardless of how the person is positioned in front of the camera.

### Webcam index fallback

Many Windows machines have the built-in webcam at index 1 (not 0) when a virtual camera or capture card is also registered. Both scripts fall back to index 1 automatically if index 0 fails to open.

---

## Phase 9 — Behavioral Biometric Innovation

### The insight

After training TCN and ST-GCN for drowsiness classification, we realised the models had learned something deeper than just "alert vs. drowsy." The penultimate layer — the 128-d global-pooled representation before the final Linear head — encodes the patterns that make each driver's behaviour unique:

- **TCN penultimate** captures the temporal signature of how a specific person's eyes, mouth, and head move
- **ST-GCN penultimate** captures the skeletal motion signature of how a specific person sits and moves their body

Two people who are both alert will have different TCN embeddings because they have different baseline EAR values, different head movement frequencies, different yawning patterns. The classifier head discards this person-specific variation to make a binary decision, but the penultimate layer retains it.

### Why this is powerful

This approach gives us **behavioral identity verification without training a new model**. The 256-d concatenated embedding is:

1. **Free** — extracted from models that already exist
2. **Temporal** — captures how a person moves, not just what they look like (harder to spoof)
3. **Complementary to ArcFace** — a face mask can fool ArcFace but cannot replicate someone's skeletal motion patterns

### Architecture surgery: adding `embed()` methods

Rather than using forward hooks (which are fragile with ONNX export), we added `embed()` methods directly to both model classes that stop before the final head:

```python
# TCN
def embed(self, x):
    out = self.net(x)        # (B, 128, W)
    out = self.pool(out)     # (B, 128, 1)
    return out.squeeze(-1)   # (B, 128)  -- penultimate

# ST-GCN
def embed(self, x):
    return self._backbone(x) # (B, 128)  -- penultimate
```

The state dictionaries load identically — only the forward pass changes.

### Data pipeline alignment

The biometric script replicates the exact data preparation pipeline from training:

| Step | Training | Biometric script |
|---|---|---|
| Face features | EAR, MAR, pitch, yaw, roll per frame | Same, per webcam frame |
| Roll clip | `clip(-90, 90)` | Same |
| Face scaling | `StandardScaler.transform()` | Same (loads `scaler.pkl`) |
| Pose shape | `(T, V, C)` → `permute(0,2,1)` → `(T, C, V)` | Same |
| TCN input | `(B, F, W)` = `(1, 5, 45)` | Same |
| ST-GCN input | `(B, T, C, V)` = `(1, 45, 4, 33)` | Same |

A mismatch at any of these steps would corrupt the embedding space and make cosine similarity meaningless.

### Verification: confirmed correct

Dry-run model loading verified shapes before deployment:
- TCN embed output: `torch.Size([1, 128])` ✓
- ST-GCN embed output: `torch.Size([1, 128])` ✓
- Unified biometric: `(256,)` ✓

---

## Deployment Challenges

### The 5 GB push failure

After committing all code, the first `git push` failed repeatedly with:

```
error: RPC failed; curl 55 Send failure: Connection was reset
```

The commit contained 41,793 PNG frames from the dataset and a 2.6 GB raw ZIP file — a 5.18 GB pack that GitHub could not receive before TCP timeouts.

**Attempted fix 1:** Increase HTTP buffer (`git config http.postBuffer 524288000`) — still failed.

**Root cause:** Even after adding `.gitignore` and `git rm --cached` for the image files, the git history retained all the objects from the initial commit. Any first push had to send the full object graph.

**Fix:** Created an orphan branch (no history) containing only the current 27 clean files, then replaced `main` with it:

```bash
git checkout --orphan fresh
git add .gitignore scripts/ data/models/ data/identity/
git commit -m "Initial commit: Ageisnet complete pipeline"
git branch -M fresh main
git push -u origin main   # succeeds: ~10 MB, not 5 GB
```

**Files excluded from git (too large or regenerable):**
- `data/processed/` — 41k PNG frames, ~5 GB (re-download from Kaggle)
- `data/raw/` — 2.6 GB dataset ZIP
- `data/features.csv`, `data/train.csv`, `data/test.csv` — regenerated by steps 2–3
- `data/pose_features.csv`, `data/pose_train.csv`, `data/pose_test.csv` — same
- `/models/` — root-level duplicate model folder

---

## Full Bug Log

| # | Bug | Script | Root cause | Fix |
|---|---|---|---|---|
| 1 | `AttributeError: module 'mediapipe' has no attribute 'solutions'` | step2 | mediapipe 0.10.35 removed `mp.solutions` | Rewrote to use Tasks API |
| 2 | 100% of frames had pitch > 90° | step2 | Euler gimbal ambiguity in `solvePnP` | Wrap: if pitch > 90 subtract 180 |
| 3 | `UnicodeEncodeError` on → arrow | step1 | Windows cp1252 terminal can't encode → | Replaced all → with -> in prints |
| 4 | `ReduceLROnPlateau` crash | step4 | PyTorch 2.10 removed `verbose` kwarg | Removed the argument |
| 5 | OOM / training never starts | pose_stgcn | stride=1 created ~33k sequences, ~780 MB | Added `STRIDE = 15` global |
| 6 | Model saved to wrong path | pose_stgcn | `ROOT/"models"` vs `ROOT/"data"/"models"` | Fixed path constant |
| 7 | Only 2,318 common test frames | step5 | Independent splits of face and pose CSVs | Inner join on normalised frame path |
| 8 | InsightFace build fails | step6 | Requires Microsoft C++ Build Tools | Switched to facenet-pytorch |
| 9 | `quantize_dynamic` crash | step7 | `optimize_model` kwarg removed in ORT 1.26 | Removed the argument |
| 10 | `KeyError: 'STGCN   FP32'` | step7 | Dict key spacing inconsistency | Explicit per-model key lookup |
| 11 | `git push` fails with curl 55 | — | 5 GB dataset committed to history | Orphan branch to squash history |

---

## Architecture Decisions That Were Changed

| What we started with | What we ended up with | Why changed |
|---|---|---|
| `mp.solutions.face_mesh` | MediaPipe Tasks API + `FaceLandmarker` | Old API removed in mediapipe 0.10.35 |
| InsightFace for ArcFace | facenet-pytorch | No C++ Build Tools on Windows |
| stride=1 for ST-GCN sequences | stride=15 | OOM crash on CPU with stride=1 |
| Separate face/pose splits | Inner-joined split | Data leakage between streams |
| `optimize_model=True` in quantisation | Removed flag | Argument removed in onnxruntime 1.26 |
| Forward hooks for penultimate layer | `embed()` method on model class | Cleaner, no hook lifecycle issues |
| Single identity method (ArcFace) | ArcFace + behavioral biometric | Two independent verification factors |

---

## Key Metrics Timeline

| Phase completed | New capability | Best accuracy achieved |
|---|---|---|
| Phase 0 | Dataset organised | — |
| Phase 1 | Face features extracted | 99.97% face detection rate |
| Phase 2 | Train/test split created | — |
| Phase 3 | TCN face classifier | **99.5%** |
| Phase 4 | ST-GCN body classifier | **98.6%** |
| Phase 5 | Fusion safety score | **99.33%** (outperforms both streams) |
| Phase 6 | ArcFace identity | 10/10, gap=0.758 |
| Phase 7 | ONNX INT8 deployment | 100.0% INT8 accuracy, 71% size reduction |
| Phase 8 | Live ArcFace webcam | Real-time registration + verification |
| Phase 9 | Behavioral biometric | 256-d embedding, no new model needed |

---

## Reflection

The biggest non-obvious insight from this project was **Phase 9**: realising that models trained for one task (drowsiness detection) had already learned representations useful for a completely different task (identity verification). The penultimate layer of a well-trained network is a general-purpose encoder of the patterns that matter for its domain — in this case, facial and skeletal behavioural dynamics.

The biggest technical lesson was the **data leakage from independent splits** discovered in Phase 5. Splitting two CSVs independently and then joining them at evaluation time silently produces train/test contamination. Any multi-modal system must split at the level of the shared unit (the frame path) from the start.

The biggest practical lesson was the **git history problem**: committing a 5 GB dataset and then trying to remove it does not shrink the repository — the objects remain in history. Always add `.gitignore` before the first commit.
