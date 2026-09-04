# Deepfake Detection — Concepts

Notes from building the deepfake image detection pipeline (Milestones M1–M4 + M7 multi-generator retraining).

## Transfer learning

A pretrained ResNet50 (ImageNet, 1000 classes) transfers to face fake/real detection even though it never saw faces. CNNs learn hierarchical features: early layers detect edges, textures, gradients (task-agnostic); later layers detect parts and semantic structures. Transfer reuses that low/mid-level visual vocabulary and adapts the top of the network to the new task.

Why it beats training from scratch: with ~100k images and 25M parameters, training from scratch needs far more data and compute to reach the same quality. Pretraining gives a strong starting representation, especially when the new dataset is modest.

## Why only the final layer changes

The input stays a 3-channel RGB image, so the first convolution and the whole feature extractor stay untouched. Only the decision layer changes: `Linear(2048, 1000)` -> `Linear(2048, 2)`. The conv stack answers "what patterns exist", the fc answers "which class". ImageNet's 1000 classes are replaced with our 2 (real/fake) — the minimum architectural change.

## Data preprocessing (input contract)

- `ToTensor()`: PIL image [0,255] HWC -> tensor [0,1] CHW.
- `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`: per-channel standardization.
  `x' = (x - mu) / sigma`.
- Why the data pipeline, not the model: normalization is preprocessing, not a learned computation. Clean separation — dataset responsibility (load, resize, tensor, normalize) vs model responsibility (extract, classify). If we swap ResNet50 for another architecture, preprocessing stays independent. Crucially, inference must apply the *same* normalization as training, else a distribution mismatch silently degrades the model.
- After standardization relative to ImageNet statistics, a given batch has mean ≈ 0, std ≈ 1 — but only approximately. Those are ImageNet's population stats, not our batch's, so divergences (e.g. mean 0.05, std 1.1) are normal and not a bug.

## Shuffling

Train `shuffle=True`: the dataset is ordered real...fake, so unshuffled batches would each contain one class, making consecutive gradients push the model the same direction — poor stochastic optimization. Shuffling makes each batch representative.

Validation/test `shuffle=False`: evaluation doesn't update weights, so order carries no signal; keeping it fixed gives determinism, reproducibility, and easy mapping of predictions back to samples.

## Freezing vs fine-tuning

- `Loss.backward()` updates only params with `requires_grad=True`. Freezing the backbone trains just the head; it preserves the pretrained representation (fast, low overfitting risk, but limited — the fixed ImageNet features may not capture deepfake-specific signals).
- Fine-tuning unfreezes everything so the hierarchy can reshape toward the new task (blending artifacts, texture inconsistency). With not enough data, the risk is overfitting or shortcut learning — "dataset/source detector" instead of a deepfake detector (e.g. memorizing Generator A's artifacts rather than general fake evidence).

Practical strategy (what we did): staged training. Stage 1 freeze backbone, train head (gets the mapping working, ~84.9% val acc). Stage 2 unfreeze all at a lower LR (lets representation adapt without destroying it, ~99.5% val acc). Use differential learning rates: new head starts random -> higher LR (1e-3); pretrained backbone is already useful -> lower LR (1e-4).

## Metrics

- **Accuracy**: OK because classes are balanced 50/50. Not sufficient alone for a detector.
- **Precision** = TP / (TP+FP): of the images we call fake, how many really are? Real-to-fake errors are false alarms.
- **Recall** = TP / (TP+FN): of the real fakes, how many did we catch? Fake-to-real errors are missed fakes — the dangerous failure for a detector.
- **F1**: harmonic mean of precision and recall (macro = averaged over classes).
- **Confusion matrix**: 2x2, shows the error cells — false alarms (FP) vs missed fakes (FN): 50 vs 71.
- **ROC-AUC**: threshold-independent, = probability a random fake scores higher than a random real. Our 0.9998 is near-perfect ranking. Since it's threshold-independent, it characterizes ranking quality, not one operating point.
- **Operating point**: the default threshold 0.5 can be tuned. If the cost of a missed fake is high, lower the threshold (more fakes found, at the cost of more false alarms on real faces). The ROC curve shows this tradeoff.

## Not a shortcut-learning detector? (answer: it was — M4 finding)

Our M4 result (99.4% accuracy, AUC 0.9998) was strong *in distribution*, but the honest caveat proved exactly right: test images come from the same two generators as training (StyleGAN fake / FFHQ real), and cross-dataset validation exposed the real generalization failure.

## The shortcut-learning lesson (M7, demonstrated with real data)

v1 was a **"StyleGAN vs FFHQ" detector, not an "AI vs real" detector**:
- Gemini-generated boy face -> predicted REAL (99.5%)
- ChatGPT-generated faces -> predicted REAL (~100%)

Why: the training set's fake class came from a single generator. The model could hit 99.4% benchmark accuracy by memorizing StyleGAN's specific artifacts (e.g. frequency signatures, blending textures) — features that don't generalize to diffusion (Midjourney/DALL-E/SD) faces. High test accuracy on a homogeneous dataset can hide total failure on generators outside the training distribution.

### Fix: multi-generator training (v2)

Train the fake class on many generation methods so the model must learn *what makes an image look machine-made*, not one generator's fingerprint:
- **Defactify (MS COCOAI, 2026)**: for each COCO caption, the real photo + 5 AI versions (SD2.1, SDXL, SD3, DALL-E 3, MidJourney v6). Because real and fake share the same scene content, the classifier cannot cheat on content ("scenes are fake, faces are real") — it must learn generation artifacts. This is the same trick as data-cleaning in tabular ML: keep the covariate distribution constant and vary only the label-driver.
- **Per-generator evaluation** via `Label_B`: after retraining, accuracy per generator on held-out test was MJ6 72%, SD2.1 72%, SD3 77%, DALL-E 3 80.5%, SDXL 85%, real 96.7% (overall 80.5%). The two hardest are the highest-fidelity generators — the residual gap tells us how close each generator is to fooling the detector.

### The generalization–memory tradeoff

Adding 5 diffusion generators to the fake class dropped in-distribution StyleGAN accuracy from 0.9940 to 0.8016. Retraining on new fakes reshapes the decision boundary toward broader "AI-ness", trading off the old generator's perfect score. This is expected and worth documenting: "99.4% then 99.9% then ..." numbers only hold when benchmark = training distribution. A detector's real quality is how it behaves on data it was never trained on (per-generator + real-world images).

Real-world verdict after v2: the same Gemini image flipped REAL (99.5%) -> FAKE (90.7%).