# Deepfake Face Detection

End-to-end deepfake image detection: fine-tuned ResNet50, wrapped in our own Streamlit app.

Two checkpoint generations:
- **v1** — trained on the 140k Real and Fake Faces dataset (FFHQ vs StyleGAN).
- **v2** — v1 fine-tuned with multi-generator data (Defactify diffusion models + DF40 face fakes), fixing a cross-generator failure discovered in real-world testing.

The model is trained from scratch on public datasets (our own pipeline) — it is not a copied demo. Data pipeline, model, training, evaluation, and app were all built and validated in this repo.

## Results

### v1 — in-distribution benchmark (held-out test: 20,000 images, StyleGAN vs FFHQ)

| metric | value |
|---|---|
| accuracy | 0.9940 |
| precision (real / fake) | 0.99 / 0.99 |
| recall (real / fake) | 0.99 / 0.99 |
| F1 (real / fake) | 0.99 / 0.99 |
| confusion matrix | TP=9929 FP=50 FN=71 TN=9950 |
| false alarm rate (real->fake) | 0.50% |
| missed fake rate | 0.71% |
| ROC-AUC | 0.9998 |

### The failure that drove v2 (shortcut learning, discovered in real-world testing)

| real-world image | v1 prediction |
|---|---|
| Gemini-generated boy face | REAL (99.5%) |
| ChatGPT-generated faces | REAL (~100%) |

v1 only ever saw StyleGAN fakes, so it learned "StyleGAN vs FFHQ" — not "AI vs real". Text-to-image diffusion faces looked real to it.

### v2 — multi-generator (per-generator accuracy on Defactify test, 45,000 images)

| source | images | accuracy |
|---|---|---|
| real (COCO photos) | 7,500 | 96.65% |
| SDXL | 7,500 | 85.15% |
| DALL-E 3 | 7,500 | 80.51% |
| SD3 | 7,500 | 76.69% |
| MidJourney v6 | 7,500 | 72.16% |
| SD2.1 | 7,500 | 71.96% |
| **ALL** | **45,000** | **80.52%** |

Generalization cost on the original benchmark: v2 scores **0.8016** on the 140k StyleGAN test (v1 was 0.9940). Cross-technique face holdout (DF40, 40 techniques): **0.8917**.

**Real-world retest with v2:**

| real-world image | v1 | v2 |
|---|---|---|
| Gemini-generated boy face | REAL (99.5%) | **FAKE (90.7%)** |

## Approach

- **v1 data**: `TheKernel01/140k-Real-and-Fake-Faces` (HuggingFace mirror of Kaggle 140k Real and Fake Faces). 70k real (FFHQ) + 70k fake (StyleGAN). Pre-split train/validation/test = 100k/20k/20k, 256x256 RGB, balanced 50/50.
- **v2 data (multi-generator)**:
  - `Rajarshi-Roy-research/Defactify_Image_Dataset` (MS COCOAI, 2026) — for each COCO caption the real photo plus 5 AI versions (SD2.1, SDXL, SD3, DALL-E 3, MidJourney v6). Caption-aligned real/fake pairs remove the "scenes vs faces" content shortcut.
  - `pujanpaudel/deepfake_face_classification` (DF40, 40 techniques) — face fakes kept as a face anchor; its `test` used as a held-out face check.
  - Balanced v2 train (8,415 real / 8,415 fake): 5,000 Defactify reals + 3,415 FFHQ faces vs 5,000 Defactify fakes (1,000 per generator) + 3,415 DF40 face fakes.
- **Preprocessing**: Resize 224, ToTensor, ImageNet Normalize; `RandomHorizontalFlip` on train only.
- **Model**: pretrained ImageNet ResNet50; last fully-connected layer replaced with `Linear(2048, 2)`.
- **Loss / optimizer**: CrossEntropyLoss, Adam with differential learning rates (backbone 1e-4, head 1e-3 for v1; 5e-5/5e-4 for the v2 stage-3 fine-tune).
- **Training**: staged (progressive unfreezing). v1: stage 1 froze the backbone (3 epochs, ~84.9% val acc), stage 2 unfroze all (2 epochs, ~99.5% val acc). v2: stage 3 continued from the v1 checkpoint on the multi-generator set (3 epochs, best val ~93.2%).

## Files

| file | purpose |
|---|---|
| `deepfake_detection.ipynb` | the full pipeline: data loading, transforms, training, evaluation (run in Google Colab) |
| `app.py` | Streamlit inference app |
| `requirements.txt` | Python dependencies for the app |

Trained checkpoints `deepfake_resnet50_v1.pt` / `deepfake_resnet50_v2.pt` (~90 MB each, stored in Google Drive) are git-ignored.

## How to run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Drop the checkpoint file `deepfake_resnet50_v2.pt` into this folder first (inference reuses the same preprocessing contract as training). The app uses v2 when present and falls back to v1.

## How to retrain

Open `deepfake_detection.ipynb` in Google Colab (T4 GPU), run all cells. The notebook downloads the datasets, builds the DataLoaders, trains both stages, evaluates (in-distribution, per-generator, real-world), and saves the checkpoints to Google Drive.