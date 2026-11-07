
from io import BytesIO
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel

from train import get_transform

st.set_page_config(page_title='FoodVision AI', page_icon='🍎', layout='wide')
st.title('🍞 FoodVision AI')
st.write('**DINOv2 Food Classification:** bread · drinks · fruit')
st.caption('Assignment 2 — Computer Vision | Upload or take a photo, predict the class, and inspect visual neighbors.')

ARTIFACTS = Path('artifacts')
DATA_DIR = Path('food_dataset')


@st.cache_resource(show_spinner='Loading DINOv2 and trained classifier...')
def load_pipeline():
    checkpoint = torch.load(ARTIFACTS / 'classifier.pt', map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = AutoModel.from_pretrained(checkpoint['model_id']).to(device).eval()
    head = nn.Linear(int(checkpoint['feature_dim']), len(checkpoint['classes']))
    head.load_state_dict(checkpoint['state_dict'])
    head = head.to(device).eval()
    data = np.load(ARTIFACTS / 'train_index.npz', allow_pickle=False)
    return encoder, head, device, checkpoint, data['embeddings'], data['paths'], data['labels']


if not (ARTIFACTS / 'classifier.pt').exists() or not (ARTIFACTS / 'train_index.npz').exists():
    st.error('Model artifacts not found. Run `python train.py --data-dir food_dataset` first.')
    st.stop()

try:
    encoder, head, device, checkpoint, train_vectors, train_paths, train_labels = load_pipeline()
except Exception as error:
    st.error('Unable to load model. Make sure `transformers` is installed and model download works.')
    st.exception(error)
    st.stop()

classes = checkpoint['classes']
left, right = st.columns([1, 1], gap='large')
with left:
    st.subheader('1. Provide a picture')
    tab_upload, tab_camera = st.tabs(['Upload', 'Camera'])
    with tab_upload:
        uploaded = st.file_uploader('Choose an image', type=['jpg','jpeg','png'])
    with tab_camera:
        camera = st.camera_input('Take a picture')
    source = uploaded if uploaded is not None else camera
    if source is None:
        st.info('Upload an image or use a camera to begin.')
    else:
        try:
            image = Image.open(BytesIO(source.getvalue())).convert('RGB')
        except (UnidentifiedImageError, OSError):
            st.error('Could not decode image. Please use a valid JPG or PNG.')
            st.stop()
        st.image(image, caption='Your input image', use_container_width=True)

with right:
    st.subheader('2. Prediction')
    if source is not None:
        # Time preprocessing + forward pass on repeated inference; model loading excluded.
        times = []
        with torch.inference_mode():
            for _ in range(5):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = perf_counter()
                x = get_transform(False)(image).unsqueeze(0).to(device)
                embedding = encoder(pixel_values=x).last_hidden_state[:, 0, :]
                logits = head(embedding)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((perf_counter() - t0) * 1000)
        pred_index = int(probs.argmax())
        label = classes[pred_index]
        emoji = {'bread':'🍞', 'drinks':'🥤', 'fruit':'🍎'}
        st.metric('Predicted class', f'{emoji.get(label, "🍽️")} {label.title()}')
        st.metric('Model confidence', f'{probs[pred_index]*100:.1f}%')
        st.metric('Avg inference time (5 runs)', f'{np.mean(times):.1f} ms')
        st.caption('Inference duration includes preprocessing and the forward pass, but not the initial model download.')
        st.bar_chart({'Class': list(classes), 'Probability': [float(v) for v in probs]}, x='Class', y='Probability', horizontal=True)

        st.subheader('3. Visual explanation: nearest neighbors')
        st.caption('These are actual training photos with similar DINOv2 visual embeddings (cosine similarity). They are NOT generated pictures. This is retrieval-based evidence, not a Grad-CAM heatmap.')
        vec = embedding.cpu().numpy().astype(np.float32)[0]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        sims = train_vectors @ vec
        neighbors = np.argsort(-sims)[:3]
        cols = st.columns(3)
        for col, idx in zip(cols, neighbors):
            path = DATA_DIR / str(train_paths[idx])
            with col:
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.info('Training image not included in this public repository.')
                st.write(f'**{classes[int(train_labels[idx])].title()}**')
                st.caption(f'Cosine similarity: {float(sims[idx]):.3f}')

st.divider()
st.subheader('Environment & limitations')
user_agent = 'Unavailable'
try:
    user_agent = st.context.headers.get('user-agent', 'Unavailable')
except Exception:
    pass
st.write(f'**Compute device:** `{device}` | **Python:** `{platform.python_version()}` | **Torch:** `{torch.__version__}`')
st.caption(f'Browser / user-agent: {user_agent}')
st.caption('Training dataset: 270 photos (90 per class). Validation: 60 photos (20 per class). The model may be wrong on unfamiliar foods, mixed meals, or images outside these three classes.')
st.caption('Data ethics: Images were gathered for an educational project. No synthetic images are created by this demo. Confirm reuse rights before publicly sharing the collected photographs.')
