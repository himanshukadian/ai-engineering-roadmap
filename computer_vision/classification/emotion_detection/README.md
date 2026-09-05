# Real-Time Emotion Detection

End-to-end FER+ emotion recognition: ResNet50 + CBAM attention, benchmarked across preprocessing variants, adapted to real photos on RAF-DB, exposed through our own Streamlit app.

**Try it live: https://resnetemotiondetection.streamlit.app/**

## Live app

Deployed on Streamlit Community Cloud. The app auto-downloads the selected checkpoint from the `emotion-v1.0.0` GitHub release on first run (default `emotion_cbam_aspect.pt`, ~97 MB).

## Model checkpoints (in `checkpoints/`)

| checkpoint | trained on | best for |
|---|---|---|
| `emotion_baseline.pt` | FER+ (M3) | benchmark baseline, plain 224 input |
| `emotion_cbam_aspect.pt` | FER+ (M4c) | webcam/grayscale-style images, default app model |
| `emotion_cbam_real.pt` | FER+ + RAF-DB (M7) | real phone/webcam-color photos |

Support files: `emotion_temps.json` (per-checkpoint temperature calibration), `emotion_centroids_cbam_real.pt` (feature-space OOD gate, real-photo model only).

## Results

### M3 baseline — ResNet50, FER+ 8-class test

| metric | value |
|---|---|
| test accuracy | 0.8626 |
| macro-F1 | 0.7269 |

### M4 CBAM ablation — same data, input pipeline as the variable

| model | accuracy | macro-F1 |
|---|---|---|
| ResNet50 baseline (plain 224) | 0.8626 | 0.7269 |
| + CBAM (plain 224) | 0.8699 | 0.7115 |
| + CBAM (face-detect crop) | 0.8677 | **0.7601** |
| + CBAM (aspect/zoom crop) | **0.8824** | 0.7488 |

Verdict: the attention module adds a small accuracy bump, but the input pipeline is the dominant knob — exactly the confounder lesson from the deepfake M7.5 work. The app's default preprocessing is aspect/zoom + CBAM.

### M7 real-photo adaptation — RAF-DB fine-tune

Motivated by a real failure: the FER+-only app read a clearly smiling real photo as HIGH neutral (90.7%). RAF-DB (20,471 in-the-wild color faces, 7 emotions) was used to adapt.

| evaluation | accuracy | macro-F1 |
|---|---|---|
| RAF-DB test (real photos, 2,048) | 0.8647 | 0.6820 |
| FER+ test (regression check) | 0.8827 | 0.7319 |

Per-class RAF test: neutral 0.99, happiness 0.92, anger 0.79, sadness 0.79, surprise 0.78, disgust 0.60, fear 0.59. The smiling-boy photo now reads HIGH happiness in the app.

## Confidence calibration (M5)

One temperature per checkpoint is fit on validation logits (NLL-minimizing softmax scaling) and applied at inference. Real-photo model: T=1.7514, ECE 0.067 -> 0.018. Verdict UI gates: multi-crop stability, confidence vs user slider, top-2 margin, and a feature-space out-of-distribution check (cosine to per-class centroids, 5th-percentile floors).

## Known limitation — subtle closed-mouth anger

The model reads certain real photos of subtle, closed-mouth anger as HIGH neutral. Two guard approaches were implemented and measured, and both are provably blind to this class:

1. **Feature-space OOD gate** — the CNN's own layers map closed-mouth anger onto its neutral manifold (cosine 0.88-0.95 to neutral, anger only 0.57-0.65), so classifier-feature distance can never flag it. The gate still fires on genuinely far inputs.
2. **Geometric second-opinion (MediaPipe landmarks)** — anger-vs-neutral LDA AUC was 0.907 within RAF-DB, but scored on real phone photos the posteriors are flat and it mislabels correct images. In-distribution confidence does not transfer.

The remaining fix is personal data: fine-tuning on a small set of the user's own labeled photos (the failure was documented with open-image geometry, not speculation). See NOTES.md for the full reasoning.

## Run the app

```
python -m venv .venv && source .venv/bin/activate   # once
pip install -r requirements.txt                     # streamlit, torch, torchvision, opencv-python-headless<5
streamlit run app.py
```

Upload a photo or use the webcam. Pick a checkpoint, the app reports a verdict (HIGH / LOW / UNSTABLE) with the gate that decided it, per-mode scores, top-3, and the scored patch.

## Repository layout

- `emotion_detection.ipynb` — the single growing notebook, M1-M5 + M7 (cells 72-77 are a self-contained cold-runtime chain for real-photo adaptation and the OOD/geometry studies)
- `core.py` — torch-only inference engine (architectures, preprocessing modes, temperatures, OOD scoring)
- `app.py` — Streamlit UI
- `checkpoints/` — models + calibration + OOD artifacts
- `PROGRESS.md` (repo root) — milestone-by-milestone audit trail