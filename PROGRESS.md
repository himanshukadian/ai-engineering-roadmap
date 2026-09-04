# Progress

## Current Status

- **Day:** 1
- **Date:** September 4, 2026
- **Phase:** 1 — Foundations

## Concepts Learned

### Mathematics
- (none yet)

### Machine Learning
- PPO (Proximal Policy Optimization): value loss, policy loss with clipping, entropy loss
- Neural networks as stacked logistic regressions
- Gradient descent in RL context (reward instead of labels)
- Training instability diagnosis via clip_fraction and approx_kl
- target_kl as early-stopping mechanism for policy updates
- explained_variance as RL equivalent of R-squared
- Statistical evaluation: sample size effect, coefficient of variation, root cause analysis

### Deep Learning
- (none yet)

### NLP
- (none yet)

### MLOps
- (none yet)

## Implementations

(none yet)

## Experiments

- `experiments/ppo_flappy_bird.ipynb` — PPO agent on FlappyBird-v0, 150k timesteps, reward from -7.6 to ~16. Environment: action_space=Discrete(2), observation_space=Box(0,1,(180,)). Diagnosed policy collapse, applied target_kl fix.

## Flagship Projects

(none yet)

## Interview Problems

(none yet)

## Benchmarks

(none yet)

## Weak Areas

To be identified as work begins.

## Deepfake Detection Project (Flagship)

Milestone-based build of a full deepfake detection pipeline (own training + own Streamlit app) inside this repo.

**M1 — Dataset setup (DONE):** Loaded `TheKernel01/140k-Real-and-Fake-Faces` (HuggingFace mirror of Kaggle 140k Real and Fake Faces). Verified balanced labels and clean splits:
- train: 100,000 (50,000 real / 50,000 fake)
- validation: 20,000 (10,000 / 10,000)
- test: 20,000 (10,000 / 10,000)
- All images 256x256 RGB. Labels: 0=real, 1=fake.

Sanity target (from published benchmark): fine-tuned ResNet50 ≈ 99.6% test accuracy.

**M2 — Data loading (DONE):** Wrapped HF splits into torch `DeepfakeDataset`; train transform = Resize(224) + RandomHorizontalFlip + ToTensor + ImageNet Normalize; eval transform = same minus flip. DataLoaders batch_size=64 (train shuffle=True, val/test shuffle=False). Sanity check passed: batch shapes (64,3,224,224), labels {0,1}, batch mean≈0, std≈1.

**M3 — Model & training (DONE):** Pretrained ResNet50 (IMAGENET1K_V1), replaced fc with Linear(2048,2). Differential LRs (backbone 1e-4, head 1e-3), CrossEntropyLoss, Adam. Staged training: Stage 1 froze backbone (3 epochs, ~84.9% val acc), Stage 2 unfroze all (2 epochs, ~99.5% val acc). Checkpoint saved to Google Drive (`deepfake_resnet50/`).

**M4 — Evaluation (DONE) on 20k test images:**
- Accuracy: 0.9940
- Precision/Recall/F1: 0.99 on both real and fake
- Confusion matrix: TP=9929 FP=50 FN=71 TN=9950
- False alarm rate (real->fake): 0.50%, missed fake rate: 0.71%
- ROC-AUC: 0.9998

**M5 — Streamlit app (DONE):** Our own `app.py` (not copied from reference). Upload -> Resize 224/ToTensor/ImageNet Normalize (same contract as training) -> ResNet50 inference -> real/fake label + softmax confidence. `requirements.txt` added; `*.pt` git-ignored; checkpoint (v1, ~90MB) copied into folder from Drive/Downloads for local demo.

**M6 — Docs (DONE):** `computer_vision/classification/README.md` (results, approach, how to run/retrain) and `NOTES.md` (ML concepts: transfer learning, fine-tuning, metrics).

**M6.5 — Real-world validation (FAILURE, the useful kind):** Tested AI images the model never trained on.
- Gemini `Gemini_Generated_Image_28mhtm28mhtm28mh.png` (AI boy) -> predicted REAL (99.5%)
- ChatGPT generated face images -> predicted REAL (~100%)

Diagnosis: **shortcut learning**. Both classes in the 140k set come from exactly two sources (FFHQ real / StyleGAN fake), so the model learned "StyleGAN vs FFHQ", not "AI vs real". High benchmark accuracy (0.9940) yet no generalization.

**M7 — Multi-generator retraining (DONE):** Fix = train fake class on many generators. Reworked M7 cells in `deepfake_detection.ipynb`:
- **DF40 mirror deficient:** `pujanpaudel/deepfake_face_classification` claimed 32,134 images, but `train.rar` extracted to only 3,415 FAKE faces (no real folder) and folders carry no per-technique names. Kept as a face anchor only (training fakes + held-out `test` eval).
- **Primary new dataset:** `Rajarshi-Roy-research/Defactify_Image_Dataset` (MS COCOAI, arXiv 2601.00553, 2026) — 96,000 images, parquet, loads with `load_dataset`. For each COCO caption there is the REAL photo + 5 AI versions: **SD2.1, SDXL, SD3, DALL-E 3, MidJourney v6**. `Label_A` 0=real/1=AI, `Label_B` per-model (0=real, 1..5). Splits train 42k / val 9k / test 45k. Caption-aligned real/fake pairs kill the content-shortcut risk and give per-generator accuracy.
- Balanced M7 train (16,830 imgs): real = 5,000 Defactify reals + 3,415 FFHQ faces (sample seed 7); fake = 5,000 Defactify fakes stratified across 5 gens (1k/gen, seed 42) + 3,415 DF40 face fakes. Perfectly balanced 8,415/8,415.
- **Stage-3 fine-tune from v1** (all layers, backbone 5e-5 / head 5e-4, 3 epochs, val=Defactify val): e1 93.2% val, e2 91.9% (overfit started), best-val weights kept.
- **Results (v2, Defactify test 45k, per-generator):** overall 80.52% — real 96.65%, SDXL 85.15%, DALL-E 3 80.51%, SD3 76.69%, MidJourney v6 72.16%, SD2.1 71.96%. **Real-world retest: Gemini boy flipped REAL(99.5%) -> FAKE(90.7%)** (verified locally too). Generalization cost: 140k StyleGAN test 0.8016 (was 0.9940); DF40 face holdout 0.8917.
- `deepfake_resnet50_v2.pt` saved to Drive, downloaded to `classification/`. `app.py` prefers v2, falls back to v1. README/NOTES updated with honest v1/v2 comparison and the shortcut-learning lesson.

## Next Actions

1. Commit and push deepfake project (M1-M7, v2)
2. Re-run Streamlit app locally to verify v2 path
3. Optional M8: push per-generator recall up (more MJ6/SD2.1 samples, more epochs, or sampling weights toward hard generators)
4. Retrain PPO with target_kl=0.05 and total_timesteps=800000
5. First learning objective: linear algebra — vectors and matrices
6. First implementation task: implement dot product, matrix multiplication, transpose from scratch
