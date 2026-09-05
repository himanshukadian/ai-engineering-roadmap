import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st
from PIL import Image

CHECKPOINT_PATH = "deepfake_resnet50_v2.pt"
FALLBACK_PATH = "deepfake_resnet50_v1.pt"

CHECKPOINT_URLS = {
    CHECKPOINT_PATH: "https://github.com/himanshukadian/ai-engineering-roadmap/releases/download/v1.0.0/deepfake_resnet50_v2.pt",
    FALLBACK_PATH: "https://github.com/himanshukadian/ai-engineering-roadmap/releases/download/v1.0.0/deepfake_resnet50_v1.pt",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["real", "fake"]

PREPROCESSING_MODES = {
    "Trained pipeline (Resize 224)": "trained",
    "Face-detect crop": "face",
    "Aspect crop (no distortion)": "aspect",
}


def ensure_checkpoint(path):
    if os.path.exists(path):
        return path
    st.info(f"Downloading {path} (~90 MB) ...")
    url = CHECKPOINT_URLS[path]
    urllib.request.urlretrieve(url, path)
    return path


@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def get_mtcnn():
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=False, device="cpu")


def trained_transform(antialias):
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=antialias),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def aspect_crop_transform(antialias):
    return transforms.Compose([
        transforms.Resize(224, antialias=antialias),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def face_crop(image, margin=0.3):
    mtcnn = get_mtcnn()
    boxes, probs = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None, False
    x1, y1, x2, y2 = [float(v) for v in boxes[0]]
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - margin * w)
    y1 = max(0, y1 - margin * h)
    x2 = min(image.width, x2 + margin * w)
    y2 = min(image.height, y2 + margin * h)
    return image.crop((int(x1), int(y1), int(x2), int(y2))), True


def preprocess(image, mode, antialias):
    if mode == "face":
        face, detected = face_crop(image)
        img = face if detected else image
        tf = transforms.Compose([
            transforms.Resize((224, 224), antialias=antialias),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return tf(img).unsqueeze(0), detected
    if mode == "aspect":
        return aspect_crop_transform(antialias)(image).unsqueeze(0), None
    return trained_transform(antialias)(image).unsqueeze(0), None


def denormalize(tensor):
    arr = tensor[0].permute(1, 2, 0).numpy()
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr, 0, 1)


def predict(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.to(device))
        prob = torch.softmax(logits, dim=1)[0]
    return prob.cpu().tolist()


@st.cache_data
def frequency_spectrum(image_bytes, size=224):
    image = Image.open(image_bytes).convert("L")
    if image.width != size or image.height != size:
        image = image.resize((size, size), Image.LANCZOS)
    gray = np.asarray(image, dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    fig, ax = plt.subplots(figsize=(3, 3), dpi=110)
    ax.imshow(magnitude, cmap="inferno")
    ax.set_title("FFT magnitude (log)")
    ax.axis("off")
    return fig


st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Face Detector")
st.caption("ResNet50 fine-tuned on real/fake faces (multi-generator: StyleGAN + DF40-40 techniques)")

try:
    model, device = load_model(ensure_checkpoint(CHECKPOINT_PATH))
except (FileNotFoundError, KeyError):
    try:
        model, device = load_model(ensure_checkpoint(FALLBACK_PATH))
    except (FileNotFoundError, KeyError):
        st.error(
            f"Checkpoint `{CHECKPOINT_PATH}` (or fallback `{FALLBACK_PATH}`) not found. "
            "Download it from Google Drive and place it in this folder."
        )
        st.stop()

with st.sidebar:
    st.header("Preprocessing")
    mode_label = st.selectbox(
        "Input pipeline",
        options=list(PREPROCESSING_MODES.keys()),
        help="The model was trained on plain Resize 224. Crop modes are experimental and may reduce accuracy.",
    )
    mode = PREPROCESSING_MODES[mode_label]
    antialias = st.checkbox("Anti-alias resize", value=True)
    show_frequency = st.checkbox("Show FFT frequency spectrum", value=False)

if mode != "trained":
    st.caption(
        "Experimental preprocessing — the model was trained on plain `Resize(224)`. "
        "Expected accuracy drop vs the default pipeline."
    )

uploaded = st.file_uploader(
    "Upload a face image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    tensor, face_detected = preprocess(image, mode, antialias)
    prob = predict(model, tensor, device)
    fake_prob = prob[1]
    pred = CLASS_NAMES[int(torch.argmax(torch.tensor(prob)))]

    col_img, col_pred = st.columns(2)
    with col_img:
        st.image(image, caption="Uploaded image", width=256)
    with col_pred:
        st.image(denormalize(tensor), caption="What the model sees", width=256)

    if mode == "face" and not face_detected:
        st.warning("No face detected — used the full image instead.")

    st.metric("Prediction", pred.upper())
    st.progress(fake_prob)
    st.write(f"Confidence — fake: {fake_prob:.2%}, real: {1 - fake_prob:.2%}")

    if show_frequency:
        uploaded.seek(0)
        st.pyplot(frequency_spectrum(uploaded))