"""
Brain Tumor AI — Interactive Web Demo
Run from project root: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from brain_tumor.config import load_config, resolve_path
from brain_tumor.inference import load_class_names, predict

cfg = load_config()
MODEL_PATH = resolve_path(cfg, "model_path")
METRICS_PATH = Path(resolve_path(cfg, "results_dir")) / "metrics.json"

st.set_page_config(
    page_title="Brain Tumor AI | Medical Imaging System",
    page_icon="MRI",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e8ecf0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #111827 100%);
    border-right: 1px solid #1e2d4a;
}
.main-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a1035 50%, #0d1f3c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
}
.main-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.main-header p { color: #94a3b8; font-size: 0.9rem; margin: 0; }
.metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
}
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
.pred-label { font-size: 2rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px; }
.pred-conf { font-family: 'Space Mono', monospace; font-size: 1rem; opacity: 0.85; }
.section-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #38bdf8;
    margin-bottom: 12px;
    font-weight: 600;
}
.disclaimer-box {
    background: #1c1d0a;
    border-left: 3px solid #ca8a04;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #a16207;
    margin-top: 16px;
}
.img-caption { font-size: 0.78rem; color: #64748b; text-align: center; margin-top: 4px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-weight:700; font-size:1.1rem; color:#38bdf8;">Brain Tumor AI</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Medical Imaging System v1.0</div>
        </div>
        <hr style="border-color:#1e2d4a; margin: 12px 0;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Settings")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help="Predictions below this confidence are flagged as uncertain.",
    )
    st.markdown("---")
    st.markdown("### Model Info")
    model_exists = os.path.exists(MODEL_PATH)
    status_color = "#22c55e" if model_exists else "#ef4444"
    status_text = "Ready" if model_exists else "Not found"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value" style="font-size:1rem; color:{status_color};">{status_text}</div>
            <div class="label">Model Status</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        st.markdown("**Validation Metrics**")
        cols = st.columns(2)
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="value" style="font-size:1.1rem;">{v:.1%}</div>
                        <div class="label">{k.replace('_', ' ').title()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown(
    """
    <div class="main-header">
        <h1>AI-Assisted Brain Tumor Detection</h1>
        <p>
            Upload a brain MRI scan. A fine-tuned EfficientNetB0 classifier predicts
            tumor presence and Grad-CAM highlights regions that influenced the decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_exists:
    st.error(
        """
        **Model weights not found.**

        Train the model first:
        ```
        python train.py --data_dir dataset/ --epochs 30 --fine_tune
        ```
        """
    )
    st.stop()

st.markdown('<div class="section-title">Upload MRI Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a brain MRI image (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.markdown(
        """
        <div style="text-align:center; padding:40px; color:#475569;">
            <div style="font-size:1rem; font-weight:500; color:#64748b;">Awaiting MRI upload</div>
            <div style="font-size:0.82rem; margin-top:8px; color:#334155;">
                Supported: JPEG, PNG | Optimal resolution: 224x224 or higher
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Running inference and generating Grad-CAM heatmap..."):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    result = predict(
        image_path=tmp_path,
        save_gradcam=True,
        gradcam_save_path=str(Path(resolve_path(cfg, "results_dir")) / "gradcam_live.png"),
        cfg=cfg,
    )
    os.unlink(tmp_path)

label = result["prediction"]
confidence = result["confidence"]
all_probs = result["all_probs"]
is_tumor = label == "Tumor"
banner_cls = "pred-tumor" if is_tumor else "pred-no-tumor"
text_color = "#fca5a5" if is_tumor else "#86efac"
uncertain = confidence < confidence_threshold
uncertain_html = (
    "<div style='font-size:0.8rem;color:#fbbf24;margin-top:8px;'>"
    "Below confidence threshold — consider manual review</div>"
    if uncertain
    else ""
)

st.markdown(
    f"""
    <div class="{banner_cls}">
        <div class="pred-label" style="color:{text_color};">{label}</div>
        <div class="pred-conf">Confidence: {confidence:.2%}</div>
        {uncertain_html}
    </div>
    """,
    unsafe_allow_html=True,
)

class_names = load_class_names(cfg=cfg)
st.markdown("**Class Probabilities**")
for cname, prob in zip(class_names, all_probs):
    disp = "Tumor" if cname.lower() in ("yes", "1", "tumor") else "No Tumor"
    bar_color = "#ef4444" if disp == "Tumor" else "#22c55e"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <div style="width:80px; font-size:0.82rem; color:#94a3b8;">{disp}</div>
            <div style="flex:1; background:#1e2d4a; border-radius:4px; height:12px;">
                <div style="width:{prob*100:.1f}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
            </div>
            <div style="width:52px; text-align:right; font-family:'Space Mono',monospace;
                        font-size:0.82rem; color:{bar_color};">{prob:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    '<div class="section-title">Explainability — Grad-CAM Visualisation</div>',
    unsafe_allow_html=True,
)
col1, col2, col3 = st.columns(3)
with col1:
    st.image(result["original_img"], use_container_width=True)
    st.markdown('<p class="img-caption">Original MRI (preprocessed)</p>', unsafe_allow_html=True)
with col2:
    st.image(result["heatmap_rgb"], use_container_width=True)
    st.markdown('<p class="img-caption">Grad-CAM Activation Map</p>', unsafe_allow_html=True)
with col3:
    st.image(result["overlay"], use_container_width=True)
    st.markdown('<p class="img-caption">Overlay — model focus region</p>', unsafe_allow_html=True)

results_dir = Path(resolve_path(cfg, "results_dir"))
st.markdown("---")
st.markdown('<div class="section-title">Training Results (last run)</div>', unsafe_allow_html=True)
res_col1, res_col2 = st.columns(2)
with res_col1:
    plot_path = results_dir / "accuracy_plot.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Training and validation curves", use_container_width=True)
    else:
        st.info("Training curves not available. Run train.py first.")
with res_col2:
    cm_path = results_dir / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Validation confusion matrix", use_container_width=True)
    else:
        st.info("Confusion matrix not available. Run train.py first.")

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Research Prototype — Not for Clinical Use.</strong>
        This system is an academic portfolio project. Results must not be used for
        medical diagnosis or treatment. Always consult a qualified neuroradiologist.
    </div>
    """,
    unsafe_allow_html=True,
)
