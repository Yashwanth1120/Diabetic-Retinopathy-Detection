"""
app.py
------
Streamlit demo: upload a retinal fundus image, get a predicted DR severity
class and a Grad-CAM heatmap showing which regions influenced the prediction.

Run with:
    streamlit run app/app.py
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model, get_target_layer          # noqa: E402
from dataset import get_transforms                        # noqa: E402
from gradcam import GradCAM, overlay_heatmap               # noqa: E402

st.set_page_config(page_title="DR Screening Demo", layout="centered")

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "best_model.pt")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    model = build_model(ckpt["backbone"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_layer = get_target_layer(model, ckpt["backbone"])
    gradcam = GradCAM(model, target_layer)

    return model, gradcam, ckpt, device


st.title("Automated Diabetic Retinopathy Detection")
st.caption("Deep learning classifier with Grad-CAM explainability")

if not os.path.exists(CHECKPOINT_PATH):
    st.warning(
        "No trained model found yet. Train one first with:\n\n"
        "`python src/train.py --data_dir data --epochs 15`\n\n"
        "then re-run this app."
    )
    st.stop()

model, gradcam, ckpt, device = load_model()
binary = ckpt["binary"]
img_size = ckpt["img_size"]
class_names = ["No DR", "DR"] if binary else \
    ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    transform = get_transforms(train=False, img_size=img_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Running prediction..."):
        cam, pred_class = gradcam.generate(input_tensor)

        # Prepare the resized original image (same size as the model input)
        # so the heatmap overlay aligns correctly.
        resized_original = np.array(image.resize((img_size, img_size)))
        overlay = overlay_heatmap(resized_original, cam)

    st.subheader(f"Prediction: {class_names[pred_class]}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_original, caption="Input (resized)", use_column_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM heatmap", use_column_width=True)

    st.info(
        "The heatmap highlights the regions of the retina that most influenced "
        "this prediction. Warmer colors (red/yellow) indicate higher importance."
    )

    st.markdown("---")
    st.caption(
        "⚠️ This is a research/educational demo, not a certified diagnostic tool. "
        "Any real screening decision should be made by a qualified ophthalmologist."
    )
