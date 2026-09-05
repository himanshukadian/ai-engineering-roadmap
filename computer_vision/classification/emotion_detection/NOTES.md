# Emotion Detection — Concepts

Notes from building the FER+ 8-class emotion recognition pipeline (Milestones M1-M5, M7).

## Attention: what CBAM adds and what it doesn't

CBAM (Convolutional Block Attention Module) computes two soft masks over a feature map:
- **Channel attention** pools each channel to a scalar (avg + max), pushes it through a tiny MLP, and sigmoids it — a learned "which channels matter" weight. `x * weight`.
- **Spatial attention** pools across channels (avg + max), runs a 7x7 convolution, sigmoids, and multiplies — a learned "which pixels matter" mask.

Both are trained end-to-end, so the masks adapt to the task. The ablation (see README) shows the honest result: CBAM gives ~+0.7pt accuracy, while choosing the right input pipeline gives ~+2.0pt. Attention improves a network; it cannot compensate for a bad input distribution. This mirrors the deepfake project's M7.5 finding: the preprocessing contract matters more than the architecture tweak.

## The preprocessing-confusion lesson, reproduced

M4 held the model and data fixed and varied only the input crop:
- Face-detect crop (Haar/MTCNN box + margin) enriched macro-F1 the most (0.7601) but hugged a noisy-class boost and dipped slightly on the stable classes.
- Aspect/zoom preserve (short side -> 256, center crop 224) gave the best accuracy (0.8824) without adding a face detector.
- Verdict written into the notebook: on aligned 48px faces, plain vs face-crop is within noise; the aspect-preserving resize is the reliably positive change.

## Temperature scaling (calibration)

A classifier can be right and still over- or under-confident. Calibration measures how much the confidence matches the empirical accuracy, via ECE (expected calibration error, 15-bin). We fit ONE scalar temperature T by minimizing the NLL of `softmax(logits / T)` on validation, and rescale at inference. For the real-photo model T=1.75 dropped ECE from 0.067 to 0.018 — the model now says "maybe" at the right rate. But calibration only rescales *confidence*; it cannot change the *label*.

## The real-photo failure and RAF-DB adaptation (M7)

The FER+-trained app read a smiling real boy as neutrality at 90.7% — confidently wrong. This is a domain gap, not a bug: FER+ is 48x48 grayscale, aligned, lab-style emotion poses; a real phone photo is color, unaligned, in-the-wild. Fine-tuning on RAF-DB (7 emotions, color, real-world) moved the decision boundary toward real photography, fixing the happiness case (0.81-0.99 in the app) without regressing FER+ (0.8827, within noise of the 0.8824 aspect model).

Why a regression check matters: the fix must not destroy the original skill. Two checkpoints were compared on the original FER+ test; the fine-tune held accuracy and lost ~0.017 macro-F1 (noise, contempt support = 0).

## Why an OOD gate cannot catch "confidently wrong" (measured negative result)

Attempt 1 — feature-space OOD. Build per-class centroids from penultimate-layer embeddings, flag images whose nearest-centroid cosine falls below a 5th-percentile floor. It works for genuinely foreign inputs (AI art, heavy crops) but is blind to the closed-mouth anger photos: those embeddings sit at cosine 0.88-0.95 from *neutral* (anger only 0.57-0.65). The network has compressed anger out of its own feature space — OOD detection in that space literally cannot represent the miss. A wrong-but-confident prediction is invisible to distance-from-data methods on the classifier's own geometry.

Attempt 2 — geometric second-opinion. Use MediaPipe face landmarks as a signal the CNN never sees: mouth aspect ratio, brow gap/height, eye openness, lip slope. Per-class LDA on 1,830 RAF-DB faces scored anger-vs-neutral AUC 0.907 (promising), but scored on the actual phone photos the posteriors flattened out (top probabilities 0.22-0.35) and it mislabeled correct images (a true angry face -> surprise, a smiling boy -> fear). The in-distribution AUC did not transfer to out-of-distribution photos. Lesson, measured twice: confidence and discrimination measured on the validation population overstate what happens on the user's own data.

The gap is a data-distribution problem, so only data can fix it: a small personal adaptation set (the user's own labeled photos) is the real remedy. See the README limitation note.

## Dataset quirks worth remembering

- FER+ has 8 classes; disgust and contempt have almost no test samples (7 and 9) — their per-class scores are noise.
- Class-weighted CE uses weights only (capped at 8x, renormalized to mean 1); combining weights with a sampler over-drove the minority classes and was dropped in the loader-bug saga.
- RAF-DB has 7 emotions (no contempt) — evaluation uses labels mapped to the FER+ 8-class space with contempt support = 0 automatically.
- Colab RAM: real-photo datasets must be lazy (decode per sample); pre-materializing 20k full-res images crashes a free runtime.