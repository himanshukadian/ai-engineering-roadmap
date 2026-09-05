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

1. Emotion M4: let M4a (CBAM plain) finish → run M4b (face-crop) → M4c (aspect) → final comparison table
2. Emotion M5: Streamlit app (image upload + webcam) using winning preprocessing mode; openCV/MediaPipe face boxes
3. Emotion M6: GitHub release (baseline + CBAM [+ variant] checkpoints) → app auto-download → live Streamlit URL → README/NOTES → commit (with approval)
4. Deepfake optional M8 (user decision): push per-generator recall (more MJ6/SD2.1 samples, more epochs, or sampling weights toward hard generators)
5. Retrain PPO with target_kl=0.05 and total_timesteps=800000
6. First learning objective: linear algebra — vectors and matrices
7. First implementation task: dot product, matrix multiplication, transpose from scratch

---

## Project: Real-Time Emotion Detection (ResNet50-CBAM)

Approach: ResNet50 with channel + spatial attention (CBAM, built from scratch in M1) after layer3/layer4, head `Linear(2048,512)->ReLU->Dropout(0.5)->Linear(512,8)`. FER+ 8 classes, CLS_NAMES order `neutral, happiness, surprise, sadness, anger, disgust, fear, contempt` (matches FER+ vote columns — verified consistent across all splits, no permutation bug). Input contract: 48×48 gray → 3ch → Resize 224 → ImageNet normalize; train aug = RandomHorizontalFlip only.

Conventions: single growing notebook `emotion_detection.ipynb` (currently 70 cells, M4 verdict appended), runs in Google Colab, user runs / agent syntax-validates only. Save dir on Drive: `/content/drive/MyDrive/emotion_detection/`.

Decisions (user): **FER+ 8-class**; **ResNet50-without-CBAM baseline** for a real ablation (the reference never measured CBAM's benefit); app = **image upload + webcam + deploy**; M4 additionally tests **preprocessing variants** (deepfake M7.5 lesson).

Data (as implemented): `chitradrishti/fer2013` raw `fer2013.csv.zip` (101MB, original `emotion,pixels,Usage` order; `load_dataset` FAILS on this repo — must download+unzip directly) joined row-aligned with `microsoft/FERPlus/fer2013new.csv` vote counts. Majority vote >50%, drop unknown/NF, split by `Usage`. Saved `ferplus_rows.pkl`. Class counts:
- train 21,811 valid (neutral 7,495 / happiness 7,083 / surprise 2,742 / sadness 2,384 / anger 1,643 / fear 340 / contempt 78 / disgust 46)
- val 2,862 valid (contempt 8, disgust 13) — test 2,729 valid (contempt 9, disgust 7)

Milestones:
- **M1 (DONE)** CBAM from scratch: ChannelAttention (avg+max pool → shared MLP r=16 → sigmoid) + SpatialAttention (concat mean/max → 7×7 conv → sigmoid) after layer3(1024) / layer4(2048). Overhead **658,822 params ≈ 2.6%** of 25.5M base. Shape/gradient/param checks pass.
- **M2 (DONE)** FER+ build + EDA above.
- **M3 (DONE) Baseline ResNet50** — results + the loader bug saga below.
- **M4 (DONE)** CBAM ablation × 3 preprocessing modes — verdict in notebook cell 70; all runs complete.
- **M5 (DONE) Streamlit app** — `app.py` (verdict UI + stability/OOD gate + temperature scaling) + `core.py` (torch-only: both architectures, 4 preprocessing modes, `analyze()` multi-crop agree/entropy/margin/face-count, `compute_centroids`/`ood_check`) + `requirements.txt` (OpenCV pinned `<5` — 5.x dropped `CascadeClassifier`). Checkpoints staged: `emotion_cbam_aspect.pt` (= renamed `emotion_cbam_ac.pt`), `emotion_baseline.pt`, `emotion_cbam_real.pt`, plus `emotion_temps.json` (T=1.7514 real model; ECE 0.067→0.018) and `emotion_centroids_cbam_real.pt` (OOD gate, active). Auto-download URLs point at the future release (M6). First-run wizard suppressed via `~/.streamlit/credentials.toml` + `--server.headless true`. **Final user-test verdict after all hardening:** boy smile → HIGH happiness (fixed vs original 90.7% neutral), b anger → HIGH anger, c/d/man-2 closed-mouth anger → known documented limitation (see negative results below).**Decision: SHIP + document** (README.md + NOTES.md written in the deepfake style).
  - **OOD gate added (M5-hardening, cells 73→76 expanded)**: c/d/man-2 test showed the model is CONFIDENTLY WRONG (softmax 1.0 neutral, even raw, no temperature can fix a wrong label) on closed-mouth anger. MediaPipe geometry probe proved it: MAR (mouth aspect ratio) 0.002–0.009 = mouths shut, identical to neutral; FER+/RAF barely contain closed-mouth anger. Fix = **feature-space OOD gate for all classes**: `core.compute_centroids/ood_check` + notebook cell 76 builds per-class 2048-d centroids on the mix train set, 5th-pct cosine threshold per class; inference `auto`-mode similarity below floor → verdict LOW regardless of softmax. App loads `checkpoints/emotion_centroids_cbam_real.pt`, gracefully skips OOD when absent.
  - **OOD gate NEGATIVE result (measured)**: c/d/man-2 sit at cosine 0.88–0.95 from the NEUTRAL centroid (anger only 0.57–0.65) — the CNN's own layers map closed-mouth anger onto its neutral manifold, so classifier-feature distance can never flag it. Gate kept (fires on genuinely-far inputs like AI art, weird crops) but documented as blind to this class.
  - **Geometry second-opinion NEGATIVE result (measured, cell 77)**: MediaPipe landmarks over 1830 RAF-val faces → 5 dims (MAR,brow_dist,brow_h,eye_open,slope); LDA 5-fold CV acc 0.510 (chance 0.283), macro-F1 0.321, anger-vs-neutral AUC **0.907 (in-dist, promising)**. BUT fitted model scored on the 5 user photos (OOD): posteriors near-flat (0.03–0.35), mislabels d (surprise 0.30), b (surprise 0.35), boy (fear 0.22>happiness 0.20) — in-dist performance does NOT transfer. Geometric guard NOT shipped (would add noise). Lesson: in-dist confidence ≠ OOD transfer; AUC on the validation population overstates usefulness for user photos.

- **M7 (DONE) real-photo adaptation** — motivated by observed failure: the app is *confidently wrong* (HIGH neutral 90.7%) on a real smiling boy photo; fix is adapting to the target distribution, not tuning the display. Notebook cells 72-75: fine-tune `emotion_cbam_ac` on **RAF-DB** (`deanngkl/raf-db-7emotions`, ~20k in-the-wild color faces, 7 emotions = FER+ minus contempt, mapped to FER+ indices) — stage-1 head-only on RAF-DB (4ep lr 1e-3), stage-2 full on FER+ + RAF-DB concat (8ep, backbone 1e-4/head 1e-3, ReduceLROnPlateau patience 2, best-val on RAF val), eval RAF test + FER+ test regression, save `emotion_cbam_real.pt` + `m7_real_results.json`, re-fit T on RAF val → merged `emotion_temps.json`. App gained the `emotion_cbam_real.pt` option. User runs cells then copies the two files into `checkpoints/`. **Results: RAF real test acc 0.8647 / macro-F1 0.6820; FER+ regression acc 0.8827 / macro-F1 0.7319 (acc preserved vs aspect model's 0.8824; macro −0.017 noise-level → no regression)**. Instead of killing frame: real-photo adsorption doc details.
  - Cell 72 is now a **fully self-contained cold-runtime bootstrap** (model classes, train helpers, transforms, FER+/RAF loaders, temp helpers) — a fresh runtime needs only cell 72/73/74; fixes rolled in: lazy `RealPhotoDataset` (RAM-crash fix), `fer_tr_loader`/`fer_te_loader` added, report uses numpy arrays + `labels=list(range(8))`, guarded `emotion_temps.json` load.
- **M6 (PENDING)** Deploy: single GitHub release `emotion-v1.0.0` (upload `emotion_baseline.pt` + `emotion_cbam_aspect.pt` ← rename of notebook's `emotion_cbam_ac.pt`), app auto-downloads to `checkpoints/`, live Streamlit URL, README/NOTES, commit (only with approval).

### M3 results (baseline, plain 224) — the record M4 must beat
- Best-val macro-F1 checkpoint. val @ s2 e8: acc 84.6% / F1 0.7107 (train 86.7%, tiny gap → no overfit).
- **Test: acc 0.8626, macro-F1 0.7269.** Per-class test F1: happiness .94, neutral .86, anger .84, fear .73, sadness .71, surprise .90, disgust .55 (support 7), contempt .29 (support 9). Disgust/contempt test support is tiny → per-class numbers for those two are ±noise only.
- Protocol: staged FT (stage-1 head-only 4ep lr 1e-3; stage-2 full 8ep backbone 1e-4/head 1e-3, ReduceLROnPlateau patience 2), class-weighted CE capped at 8× (below), no sampler, seed 42.

### THE loader bug saga — DO NOT reintroduce
First M3 runs collapsed: val acc 1.8-3% (chance = 12.5%), f1 ≈ 0.02, train acc stuck ~27-37%, train loss falling. Identical signature across 3 attempts. Root cause = **double-overdrive**:
- Weighted CE with raw `w_c = N/(8·n_c)` renormalized to mean 1 → disgust weight ≈ 4.44 (i.e. 59× raw) — AND `WeightedRandomSampler` ALSO oversampling rare classes (~148× a neutral row). Weights+sampler together made the head predict rare classes almost always → catastrophically confident-wrong on val.
- Reusable diagnostic: (1) print `type(train_loader.sampler)` + `criterion.weight`; (2) **untrained-head probe** = fresh model, zero training, eval on val → healthy ≈ 12.5% acc / ln8=2.08 loss; broken feed ≈ 1-2% acc / ~2.9 loss. Probe confirmed images+lables healthy (no permutation bug).
- Fix applied: **weights only, capped** `w_c = min(N/(8·n_c), 8)`, mean-renormalized → `[0.10, 0.11, 0.28, 0.32, 0.47, 2.24, 2.24, 2.24]`; plain shuffled loaders; fresh seed-42 model. Immediate result: s1 e1 val **45.4%** (was 1.8%). Notebook patched (weights cell + imbalance md); PIL `mode='L'` deprecation silenced.

### M4 — preprocessing confounder ablation (deepfake M7.5 lesson applied)
Same M3 protocol verbatim; only variable = model/preprocessing. Fresh seed-42 model each. Each run saves its `.pt` + results JSON; final cell builds the comparison table:
- **M3** baseline plain 224 → `m3_baseline_results.json` (DONE 0.8626 / 0.7269)
- **M4a** CBAM plain 224 → `m4_cbam_results.json` (DONE 0.8699 / 0.7115)
- **M4b** CBAM + MTCNN face-detect crop (15% margin; boxes precomputed once → Drive `ferplus_face_boxes.pkl`) → `m4b_fd_results.json`
- **M4c** CBAM + aspect/zoom crop (256 → center 224 ≈ 14% zoom) → `m4c_aspect_results.json`

Expectation: M4c ≈ M4a within noise (square input makes aspect-crop degenerate — measured "no change" IS the result). **M4a status: DONE** — best-val F1 0.7283/87.1% (e7) vs baseline 0.7107/84.6%. Test: **acc 0.8699 (+0.0073) / macro-F1 0.7115 (−0.0153)**. CBAM wins acc; macro-F1 "loss" is entirely ONE noise cell (disgust −0.260 on 7 support) — excluding it CBAM is +0.017 macro-F1, biggest on fear +0.035 / contempt +0.078 / neutral +0.019 / anger +0.016. Checkpoint size delta (emotion_cbam.pt 101.2MB − baseline 98.6MB = +2.6MB) independently confirms the M1 overhead math (658,822 params). Winning preprocessing mode becomes the M5 app default.

**M4b status: DONE (CBAM + face-detect crop)** — Haar/MTCNN box + 15% margin; boxes cached `ferplus_face_boxes.pkl`. Stage-1 started below baseline (tight crop removes context on aligned faces — mirrors deepfake M7.5) but recovered; **test acc 0.8677 / macro-F1 0.7601 (−0.002 acc, +0.033 macro vs baseline; best macro so far)**. Caveat: disgust 0.83 (was 0.29) on 7 support drives much of the macro gain; on the 6 stable classes plain CBAM slightly edges face-crop (0.841 vs 0.825). Conclusion forming: on aligned faces crop ≈ plain within noise.

**M4c status: DONE (CBAM + aspect/zoom crop)** — Resize 256 → center-crop 224 (~14% zoom ≈ mild upscale boost vs plain 48→224). **test acc 0.8824 (+0.020 vs baseline; BEST acc) / macro-F1 0.7488 (+0.022)**. Notably the preprocessing knob (+2.0pt acc) matters MORE than CBAM itself (+0.7pt) — the deepfake preprocessing-confounder lesson reproduces. Stable-6 class macro: baseline 0.830 / M4a 0.841 / M4b 0.825 / M4c 0.838 — all CBAMs ≈ baseline within noise there; CBAM's honest gain shows on subtle classes (fear/neutral/anger). Final table cell (68) aggregates.
