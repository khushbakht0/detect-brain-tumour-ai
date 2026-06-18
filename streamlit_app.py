"""
streamlit_app.py — Brain Tumor AI: Interactive Web Demo
========================================================
A polished Streamlit interface for uploading brain MRI images and receiving
AI-powered predictions with Grad-CAM explainability.

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import json
import tempfile
import numpy as np
from PIL import Image

import streamlit as st

# Resolve imports when run from app/ or project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import predict, load_model, load_class_names


# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor AI | Medical Imaging System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────
# Custom CSS — dark medical theme
# ──────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Base ─────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e8ecf0;
}

/* ── Sidebar ─────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #111827 100%);
    border-right: 1px solid #1e2d4a;
}

/* ── Main header ─────────────── */
.main-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a1035 50%, #0d1f3c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0;
}

/* ── Metric cards ─────────────── */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* ── Prediction banner ────────── */
.pred-tumor {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1.5px solid #ef4444;
    border-radius: 10px;
    padding: 20px 28px;
    text-align: center;
    margin: 12px 0;
}
.pred-no-tumor {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1.5px solid #22c55e;
    border-radius: 10px;
    padding: 20px 28px;
    text-align: center;
    margin: 12px 0;
}
.pred-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.pred-conf {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    opacity: 0.85;
}

/* ── Section cards ────────────── */
.section-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #38bdf8;
    margin-bottom: 12px;
    font-weight: 600;
}

/* ── Upload zone ──────────────── */
[data-testid="stFileUploader"] {
    background: #0d1425 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
}

/* ── Progress / spinner ───────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
}

/* ── Image captions ───────────── */
.img-caption {
    font-size: 0.78rem;
    color: #64748b;
    text-align: center;
    margin-top: 4px;
}

/* ── Warning / info boxes ─────── */
.disclaimer-box {
    background: #1c1d0a;
    border-left: 3px solid #ca8a04;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #a16207;
    margin-top: 16px;
}

/* ── Scrollbar ────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:2.4rem">🧠</div>
        <div style="font-weight:700; font-size:1.1rem; color:#38bdf8;">Brain Tumor AI</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Medical Imaging System v1.0</div>
    </div>
    <hr style="border-color:#1e2d4a; margin: 12px 0;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99, value=0.70, step=0.01,
        help="Predictions below this confidence are flagged as uncertain."
    )

    gradcam_alpha = st.slider(
        "Grad-CAM blend intensity",
        min_value=0.2, max_value=0.8, value=0.45, step=0.05,
        help="Controls how strongly the heatmap is overlaid on the MRI."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    model_exists = os.path.exists("models/brain_tumor_model.h5")
    status_color = "#22c55e" if model_exists else "#ef4444"
    status_text  = "Loaded" if model_exists else "Not found"

    st.markdown(f"""
    <div class="metric-card">
        <div class="value" style="font-size:1rem; color:{status_color};">
            {'✓' if model_exists else '✗'} {status_text}
        </div>
        <div class="label">Model Status</div>
    </div>
    """, unsafe_allow_html=True)

    # Load metrics if available
    metrics_path = "results/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.markdown("**Validation Metrics**")
        cols = st.columns(2)
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value" style="font-size:1.1rem;">{v:.1%}</div>
                    <div class="label">{k.replace('_', ' ').title()}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; text-align:center; line-height:1.5;">
        Built with EfficientNetB0 + Grad-CAM<br>
        Transfer learning on MRI dataset<br><br>
        <span style="color:#1e3a5f;">FAST-NU CS · Medical AI Portfolio</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 AI-Assisted Brain Tumor Detection</h1>
    <p>
        Upload a T1-weighted MRI scan below. The system uses a fine-tuned
        EfficientNetB0 neural network to classify the scan and generates
        Grad-CAM heatmaps to show which regions influenced the decision.
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Check model availability
# ──────────────────────────────────────────────
if not os.path.exists("models/brain_tumor_model.h5"):
    st.error("""
    ⚠️ **Model weights not found.**

    Train the model first by running:
    ```
    python train.py --data_dir dataset/ --epochs 30 --fine_tune
    ```
    The model will be saved to `models/brain_tumor_model.h5` automatically.
    """)
    st.stop()


# ──────────────────────────────────────────────
# Upload section
# ──────────────────────────────────────────────
st.markdown('<div class="section-title">📤 Upload MRI Image</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a brain MRI image here (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
    help="Upload a T1-weighted axial MRI slice for tumor detection."
)


# ──────────────────────────────────────────────
# Example images notice
# ──────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:40px; color:#475569;">
        <div style="font-size:3.5rem; margin-bottom:12px;">🔬</div>
        <div style="font-size:1rem; font-weight:500; color:#64748b;">
            Awaiting MRI upload
        </div>
        <div style="font-size:0.82rem; margin-top:8px; color:#334155;">
            Supported formats: JPEG · PNG  |  Optimal resolution: 224×224 or higher
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────
# Run inference
# ──────────────────────────────────────────────
with st.spinner("Running inference and generating Grad-CAM heatmap..."):
    # Save upload to a temp file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Run full prediction pipeline
    result = predict(
        image_path=tmp_path,
        save_gradcam=True,
        gradcam_save_path=os.path.join("results", f"gradcam_live.png"),
    )
    os.unlink(tmp_path)  # clean up temp file

label      = result["prediction"]
confidence = result["confidence"]
all_probs  = result["all_probs"]
overlay    = result["overlay"]
heatmap    = result["heatmap_rgb"]
original   = result["original_img"]


# ──────────────────────────────────────────────
# Prediction banner
# ──────────────────────────────────────────────
is_tumor   = label == "Tumor"
banner_cls = "pred-tumor" if is_tumor else "pred-no-tumor"
emoji      = "🔴" if is_tumor else "🟢"
text_color = "#fca5a5" if is_tumor else "#86efac"
uncertain  = confidence < confidence_threshold

st.markdown(f"""
<div class="{banner_cls}">
    <div class="pred-label" style="color:{text_color};">{emoji} {label}</div>
    <div class="pred-conf">Confidence: {confidence:.2%}</div>
    {"<div style='font-size:0.8rem;color:#fbbf24;margin-top:8px;'>⚠️ Below confidence threshold — consider manual review</div>" if uncertain else ""}
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Probability bars
# ──────────────────────────────────────────────
class_names = load_class_names()
st.markdown("**Class Probabilities**")
for i, (cname, prob) in enumerate(zip(class_names, all_probs)):
    disp = "Tumor" if cname.lower() in ("yes", "1", "tumor") else "No Tumor"
    bar_color = "#ef4444" if disp == "Tumor" else "#22c55e"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <div style="width:80px; font-size:0.82rem; color:#94a3b8;">{disp}</div>
        <div style="flex:1; background:#1e2d4a; border-radius:4px; height:12px; overflow:hidden;">
            <div style="width:{prob*100:.1f}%; height:100%;
                        background:{bar_color}; border-radius:4px;
                        transition: width 0.6s ease;"></div>
        </div>
        <div style="width:52px; text-align:right; font-family:'Space Mono',monospace;
                    font-size:0.82rem; color:{bar_color};">{prob:.2%}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ──────────────────────────────────────────────
# Image panels: Original | Grad-CAM | Overlay
# ──────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Explainability — Grad-CAM Visualisation</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(original, use_container_width=True, clamp=True)
    st.markdown('<p class="img-caption">Original MRI (preprocessed)</p>',
                unsafe_allow_html=True)

with col2:
    st.image(heatmap, use_container_width=True, clamp=True)
    st.markdown('<p class="img-caption">Grad-CAM Activation Map</p>',
                unsafe_allow_html=True)

with col3:
    st.image(overlay, use_container_width=True, clamp=True)
    st.markdown('<p class="img-caption">Overlay — model focus region</p>',
                unsafe_allow_html=True)

st.markdown("""
<div style="background:#0d1f3c; border:1px solid #1e3a5f; border-radius:8px;
            padding:14px 18px; margin-top:12px; font-size:0.82rem; color:#94a3b8;">
    <strong style="color:#38bdf8;">How to read this:</strong>
    The Grad-CAM heatmap highlights regions the network considers most diagnostic.
    <span style="color:#ef4444;">Red/warm regions</span> indicate high activation (areas driving
    the prediction). <span style="color:#22c55e;">Cool regions</span> are less influential.
    For a <em>Tumor</em> classification, warm regions should concentrate around
    the mass or abnormal tissue.
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Additional results artefacts
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📈 Training Results (from last run)</div>',
            unsafe_allow_html=True)

res_col1, res_col2 = st.columns(2)
accuracy_plot = "results/accuracy_plot.png"
confusion_mat = "results/confusion_matrix.png"

with res_col1:
    if os.path.exists(accuracy_plot):
        st.image(accuracy_plot, caption="Training & Validation Accuracy / Loss",
                 use_container_width=True)
    else:
        st.info("Training curves not yet available. Run train.py first.")

with res_col2:
    if os.path.exists(confusion_mat):
        st.image(confusion_mat, caption="Confusion Matrix on Validation Set",
                 use_container_width=True)
    else:
        st.info("Confusion matrix not yet available. Run train.py first.")


# ──────────────────────────────────────────────
# Disclaimer
# ──────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Research Prototype — Not for Clinical Use.</strong>
    This system is developed as an academic/portfolio project.
    Results must not be used for medical diagnosis or treatment decisions.
    Always consult a qualified neuroradiologist for clinical evaluation.
</div>
""", unsafe_allow_html=True)
