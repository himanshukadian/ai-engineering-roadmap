import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st
from PIL import Image

CHECKPOINT_PATH = "deepfake_resnet50_v2.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["real", "fake"]


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


def predict(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0]
    return prob.cpu().tolist()


st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Face Detector")
st.caption("ResNet50 fine-tuned on real/fake faces (multi-generator: StyleGAN + DF40-40 techniques)")

try:
    model, device = load_model(CHECKPOINT_PATH)
except FileNotFoundError:
    fallback = "deepfake_resnet50_v1.pt"
    try:
        model, device = load_model(fallback)
    except FileNotFoundError:
        st.error(
            f"Checkpoint `{CHECKPOINT_PATH}` (or fallback `{fallback}`) not found. "
            "Download it from Google Drive and place it in this folder."
        )
        st.stop()

uploaded = st.file_uploader(
    "Upload a face image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=256)

    prob = predict(model, image, device)
    fake_prob = prob[1]
    pred = CLASS_NAMES[int(torch.argmax(torch.tensor(prob)))]

    st.metric("Prediction", pred.upper())
    st.progress(fake_prob)
    st.write(f"Confidence — fake: {fake_prob:.2%}, real: {1 - fake_prob:.2%}")