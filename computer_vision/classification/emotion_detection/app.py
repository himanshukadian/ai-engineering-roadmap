import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import core

MODELS = {
    'CBAM + aspect/zoom (best)': ('cbam', 'emotion_cbam_aspect.pt', ['auto', 'aspect', 'plain']),
    'CBAM + RAF-DB real-photo (real photos)': ('cbam', 'emotion_cbam_real.pt', ['auto', 'aspect', 'plain']),
    'Baseline ResNet50': ('plain', 'emotion_baseline.pt', ['plain']),
}

RELEASE_BASE = (
    'https://github.com/himanshukadian/ai-engineering-roadmap/releases/download/'
    'emotion-v1.0.0/'
)
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

st.set_page_config(page_title='Real-Time Emotion Detection', layout='centered')


def load_temps():
    path = os.path.join(SAVE_DIR, 'emotion_temps.json')
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


TEMPS = load_temps()

CENTROID_FILES = {'emotion_cbam_real.pt': 'emotion_centroids_cbam_real.pt'}


def load_centroids(ckpt_name):
    if ckpt_name not in CENTROID_FILES:
        return None
    path = os.path.join(SAVE_DIR, CENTROID_FILES[ckpt_name])
    if not os.path.exists(path):
        return None
    try:
        data = torch_load_centroids(path)
        return data['centroids'], data['thresholds']
    except Exception:
        return None


def torch_load_centroids(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


@st.cache_resource
def get_model(kind, name):
    path = ensure_checkpoint(name)
    return core.load_checkpoint(core.build_model(kind), path, 'cpu')


def ensure_checkpoint(name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name)
    if os.path.exists(path):
        return path
    url = RELEASE_BASE + name
    with st.spinner('Downloading ' + name + ' (~100 MB, first run only) ...'):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                'Could not download the checkpoint. For local use, copy the .pt '
                'files into the checkpoints/ folder next to the app.'
            ) from exc
    return path


def verdict(a, threshold, ood=None):
    top1conf = a['top3'][0][1]
    stability_gate = len(a['rows']) > 1 and not a['stable']
    margin_gate = a['margin'] < 0.10
    conf_gate = top1conf < threshold

    if stability_gate:
        return 'UNSTABLE', (
            'Crops disagree on the top emotion ({}). This looks out-of-distribution '
            'for a 48px-aligned-face model (generated art, group shot, heavy pose/'
            'lighting). Label below is unreliable.'
        ).format('/'.join(sorted(a['agreement'])))
    if ood:
        cls, sim = ood
        return 'LOW', (
            'OUT-OF-DISTRIBUTION for {}: nearest training-class fit is only {:.1%} '
            '(below the {:.1%} in-distribution floor). Softmax confidence is '
            'meaningless here -- the image is not like anything the model was trained on.'
        ).format(cls, sim, sim)
    if not (margin_gate or conf_gate):
        return 'HIGH', (
            'Stable across crops with {:.1%} confidence and a {:.2f} margin '
            '(above your {:.0%} slider).'
        ).format(top1conf, a['margin'], threshold)

    reasons = []
    if conf_gate:
        reasons.append('{:.1%} < slider {:.0%}'.format(top1conf, threshold))
    if margin_gate:
        reasons.append('margin {:.2f} < 0.10 (two close candidates)'.format(a['margin']))
    return 'LOW', (
        'Fails the confidence gate: {}. The {:.1%} label is not trustworthy enough '
        'for a high-confidence read.'
    ).format(' and '.join(reasons), top1conf)


def run_inference(model, mode, file):
    img = Image.open(io.BytesIO(file.getvalue())).convert('RGB')
    temp = float(TEMPS.get(ckpt_name, {}).get('T', 1.0))

    c1, c2 = st.columns(2)
    with c1:
        preview = img.copy()
        if mode in ('auto', 'face') and core.face_boxes(img):
            draw = ImageDraw.Draw(preview)
            for (x, y, w, h) in core.face_boxes(img):
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=4)
        st.image(preview, caption='Input with detected faces (green = scored face)', width=280)
    with c2:
        patch = core.preprocess_pil(img, mode if mode is not None else 'auto')
        st.image(patch, caption='Patch the model actually scored ({}x{})'.format(*patch.size), width=280)

    a = core.analyze(img, model, model_modes, 'cpu', temp)

    centroids = None
    if not is_baseline:
        centroids = load_centroids(ckpt_name)
    ood = None
    if centroids is not None:
        idx, sim, all_sims, is_ood = core.ood_check(model, img, 'auto', centroids[0], centroids[1], 'cpu')
        if is_ood:
            ood = (core.CLS_NAMES[idx], sim)

    label, note = verdict(a, threshold, ood)
    color = {'HIGH': 'green', 'LOW': 'orange', 'UNSTABLE': 'red'}[label]

    top = a['top3'][0]
    st.markdown(
        '### <span style="color:{}">{} : {}</span>  <small>({:.1%})</small>'.format(
            color, label, top[0], top[1]
        ),
        unsafe_allow_html=True,
    )
    st.write(note)

    if a['n_faces'] > 1:
        st.warning('{} faces detected -- scoring the largest one.'.format(a['n_faces']))

    cols = st.columns(3)
    for col, (name, prob) in zip(cols, a['top3']):
        col.metric(name, '{:.1%}'.format(prob))

    mode_table = pd.DataFrame(
        {
            'mode': [r['mode'] for r in a['rows']],
            'top': [r['top'] for r in a['rows']],
            'conf': ['{:.1%}'.format(r['conf']) for r in a['rows']],
        }
    )
    st.table(mode_table)

    st.bar_chart(pd.Series(a['rows'][0]['probs'], index=core.CLS_NAMES))
    st.caption('Temperature {:.3f} (calibrated on FER+ val)'.format(temp))
    if ood is not None:
        st.caption('Nearest training-rep fit: {:.1%} ({}) -- outside the in-distribution floor'.format(ood[1], ood[0]))


with st.sidebar:
    st.header('Settings')
    model_name = st.selectbox('Checkpoint', list(MODELS.keys()))
    kind, ckpt_name, model_modes = MODELS[model_name]
    is_baseline = kind == 'plain'
    if is_baseline:
        st.selectbox('Preprocessing', ['Plain 224 (baseline fixed)'], disabled=True)
        mode = 'plain'
    else:
        mode = st.selectbox('Preprocessing', ['Auto (face detect)', 'Aspect/zoom', 'Face-detect crop', 'Plain 224'])
        mode = {'Auto (face detect)': 'auto', 'Aspect/zoom': 'aspect',
                'Face-detect crop': 'face', 'Plain 224': 'plain'}[mode]
    threshold = st.slider('High-confidence threshold', 0.30, 0.95, 0.60, 0.05)

model = get_model(kind, ckpt_name)

tabs = st.tabs(['Webcam', 'Upload'])
with tabs[0]:
    shot = st.camera_input('Take a picture')
    if shot is not None:
        run_inference(model, mode, shot)
with tabs[1]:
    uploaded = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded is not None:
        run_inference(model, mode, uploaded)