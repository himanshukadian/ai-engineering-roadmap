import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

CLS_NAMES = [
    'neutral', 'happiness', 'surprise', 'sadness',
    'anger', 'disgust', 'fear', 'contempt',
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
FINAL = 224


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        attn = self.mlp(avg) + self.mlp(max_)
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResNet50_Base(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        backbone = models.resnet50(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def features(self, x):
        """2048-d penultimate embedding (before the classifier head)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if hasattr(self, 'cbam3'):
            x = self.cbam3(x)
        x = self.layer4(x)
        if hasattr(self, 'cbam4'):
            x = self.cbam4(x)
        return self.avgpool(x).flatten(1)


class ResNet50_Plain(ResNet50_Base):
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x)


class ResNet50_CBAM(ResNet50_Base):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        return self.classifier(x)


def build_model(kind='cbam'):
    if kind == 'plain':
        return ResNet50_Plain(num_classes=8)
    return ResNet50_CBAM(num_classes=8)


def load_checkpoint(model, path, device='cpu', weights_only=True):
    state = torch.load(path, map_location=device, weights_only=weights_only)
    model.load_state_dict(state)
    model.eval()
    return model


def _face_detector():
    import cv2
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def face_boxes(img):
    import cv2
    gray = np.array(img.convert('L'))
    boxes = _face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]


def _largest_face_box(img):
    boxes = face_boxes(img)
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def _crop_to_face_pil(img, margin=0.15):
    box = _largest_face_box(img)
    if box is None:
        return img
    x, y, w, h = box
    m = int(max(w, h) * margin)
    x1, y1 = max(0, x - m), max(0, y - m)
    x2, y2 = min(img.width, x + w + m), min(img.height, y + h + m)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _plain_pil(img):
    return img.resize((FINAL, FINAL), Image.BICUBIC)


def _face_pil(img):
    return _crop_to_face_pil(img).resize((FINAL, FINAL), Image.BICUBIC)


def _aspect_pil(img, short=256):
    w, h = img.size
    scale = short / min(w, h)
    nw = max(FINAL, round(w * scale))
    nh = max(FINAL, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - FINAL) // 2
    top = (nh - FINAL) // 2
    return img.crop((left, top, left + FINAL, top + FINAL))


def _auto_pil(img):
    if _largest_face_box(img) is not None:
        return _face_pil(img)
    return _aspect_pil(img)


_PREPROCESSORS = {
    'plain': _plain_pil,
    'aspect': _aspect_pil,
    'face': _face_pil,
    'auto': _auto_pil,
}


def preprocess_pil(img, mode):
    return _PREPROCESSORS[mode](img)


def _to_input(pil):
    x = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])(pil)
    return x.unsqueeze(0)


def compute_centroids(model, loader, device='cpu', num_classes=8):
    """Per-class feature centroids + per-class cosine similarities over a loader."""
    model.eval()
    feats = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for images, labels in loader:
            f = model.features(images.to(device)).cpu()
            for i, label in enumerate(labels):
                feats[label.item()].append(f[i])
    centroids = []
    for c in range(num_classes):
        if feats[c]:
            centroids.append(torch.stack(feats[c]).mean(0))
        else:
            centroids.append(torch.zeros(1))
    centroids = torch.stack(centroids)
    norm_c = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sims = {c: [] for c in range(num_classes)}
    for c in range(num_classes):
        if not feats[c]:
            continue
        stack = torch.stack(feats[c])
        norm_s = stack / stack.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sims[c] = (norm_s @ norm_c[c]).tolist()
    return centroids, sims


def ood_check(model, img, mode, centroids, thresholds, device='cpu'):
    """OOD gate: cosine similarity of the image embedding to each class centroid.
    Returns (nearest_idx, nearest_sim, all_sims, ood_flag)."""
    model.eval()
    x = _to_input(preprocess_pil(img, mode))
    with torch.no_grad():
        feat = model.features(x.to(device)).cpu()[0]
    norm_f = feat / feat.norm().clamp_min(1e-12)
    norm_c = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sims = (norm_f @ norm_c.t()).numpy()
    idx = int(sims.argmax())
    ood = float(sims[idx]) < thresholds[idx]
    return idx, float(sims[idx]), sims, ood


def predict(model, img, mode, device='cpu', temp=1.0):
    x = _to_input(preprocess_pil(img, mode))
    with torch.no_grad():
        logits = model(x.to(device)).cpu()[0]
        if temp != 1.0:
            logits = logits / temp
        probs = torch.softmax(logits, dim=0).numpy()
    return probs, CLS_NAMES[int(probs.argmax())]


def analyze(img, model, modes, device='cpu', temp=1.0):
    rows = []
    for mode in modes:
        probs, top = predict(model, img, mode, device=device, temp=temp)
        rows.append({'mode': mode, 'probs': probs, 'top': top, 'conf': float(probs.max())})
    primary = rows[0]
    top1s = {r['top'] for r in rows}
    stable = len(top1s) == 1
    order = np.argsort(primary['probs'])[-3:][::-1]
    top3 = [(CLS_NAMES[i], float(primary['probs'][i])) for i in order]
    p = primary['probs']
    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - sorted_p[1])
    eps = 1e-12
    entropy = float(-(p * np.log(p + eps)).sum() / np.log(len(CLS_NAMES)))
    return {
        'rows': rows,
        'stable': stable,
        'agreement': sorted(top1s),
        'top3': top3,
        'margin': margin,
        'entropy': entropy,
        'n_faces': len(face_boxes(img)),
    }