"""
Brain Tumor AI — Clinical Imaging Platform
Run from project root: streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import numpy as np
import base64
import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from brain_tumor.config import load_config, resolve_path
from brain_tumor.inference import load_class_names, predict
from report_generator import AI_SYSTEM_VERSION, build_report_from_prediction
from utils.report_utils import assess_risk, load_validation_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()
MODEL_PATH = resolve_path(cfg, "model_path")
RESULTS_DIR = Path(resolve_path(cfg, "results_dir"))
METRICS_PATH = RESULTS_DIR / "metrics.json"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor AI | Clinical Imaging Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — minimal clinical, off-white, boxes on scans only
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #F5F4F0;
    --bg-warm: #FAF9F6;
    --ink: #0F172A;
    --muted: #64748B;
    --primary: #2563EB;
    --border: #D4D4CE;
    --danger: #EF4444;
    --success: #22C55E;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--ink);
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section {
    background: var(--bg) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-warm) !important;
    border-right: 1px solid var(--border);
}

/* ---- Typography blocks (no boxes) ---- */
.page-title {
    margin: 0 0 6px 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ink);
}

.page-subtitle {
    margin: 0 0 28px 0;
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 680px;
    line-height: 1.55;
}

.section-heading {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 28px 0 14px 0;
}

.section-heading i { margin-right: 8px; color: var(--primary); font-size: 0.75rem; }

/* ---- Flat stats (dashboard, no boxes) ---- */
.stat-flat { margin-bottom: 8px; }
.stat-flat .stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 4px;
}
.stat-flat .stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.1;
}
.stat-flat .stat-sub { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }

/* ---- Detection result (no box) ---- */
.result-block { margin: 8px 0 24px 0; padding: 0; }
.result-eyebrow {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin-bottom: 8px;
    font-weight: 600;
}
.result-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 10px 0;
}
.result-title.tumor { color: var(--danger); }
.result-title.clear { color: var(--success); }
.result-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem;
    color: var(--muted);
}
.result-note {
    font-size: 0.82rem;
    color: #854D0E;
    margin-top: 10px;
}

/* ---- Probability bars (no box) ---- */
.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
}
.prob-label { width: 88px; font-size: 0.85rem; color: var(--muted); }
.prob-track {
    flex: 1;
    background: #E8E6E1;
    border-radius: 999px;
    height: 6px;
}
.prob-fill { height: 100%; border-radius: 999px; }
.prob-pct {
    width: 52px;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}

/* ---- MRI / scan frames ONLY ---- */
.scan-frame {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px;
    background: var(--bg-warm);
    margin-bottom: 4px;
}
.scan-frame img {
    width: 100%;
    display: block;
    border-radius: 6px;
}
.img-caption {
    font-size: 0.75rem;
    color: var(--muted);
    text-align: center;
    margin-top: 8px;
}

/* ---- Plain text helpers ---- */
.plain-note {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.55;
    margin: 8px 0 16px 0;
}
.empty-state {
    text-align: center;
    padding: 48px 16px;
    color: var(--muted);
}
.empty-state .empty-title {
    font-weight: 600;
    color: #475569;
    margin-top: 12px;
}
.empty-state .empty-sub { font-size: 0.85rem; margin-top: 8px; }

.disclaimer-text {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    line-height: 1.55;
}

.status-line {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 8px;
}
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}

div[data-testid="stDownloadButton"] button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Hide default file uploader heavy chrome slightly */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", icon: str = "fa-chart-line", color: str = "#0F172A"):
    st.markdown(
        f"""
        <div class="stat-flat">
            <div class="stat-label"><i class="fa-solid {icon}" style="margin-right:6px;"></i>{title}</div>
            <div class="stat-value" style="color:{color};">{value}</div>
            {"<div class='stat-sub'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, icon: str = "fa-circle"):
    st.markdown(
        f'<div class="section-heading"><i class="fa-solid {icon}"></i>{title}</div>',
        unsafe_allow_html=True,
    )


def render_scan_image(arr, caption: str) -> None:
    """Render MRI with the only bordered frame in the UI."""
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f"""
        <div class="scan-frame">
            <img src="data:image/png;base64,{b64}" alt="{caption}"/>
            <p class="img-caption">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "scan_filename" not in st.session_state:
    st.session_state.scan_filename = None
if "report_bytes" not in st.session_state:
    st.session_state.report_bytes = None
if "report_name" not in st.session_state:
    st.session_state.report_name = None

model_exists = os.path.exists(MODEL_PATH)
validation_metrics = load_validation_metrics(METRICS_PATH)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:12px 0 4px 0;">
            <div style="font-size:1.15rem; font-weight:700; color:#0F172A;">
                <i class="fa-solid fa-brain" style="color:#2563EB; margin-right:8px;"></i>
                Brain Tumor AI
            </div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:4px;">
                Clinical Imaging Platform v{AI_SYSTEM_VERSION}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "MRI Analysis", "Reports"],
        icons=["speedometer2", "file-medical", "file-earmark-pdf"],
        default_index=1,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#2563EB", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "500",
                "color": "#475569",
                "margin": "4px 0",
                "border-radius": "10px",
            },
            "nav-link-selected": {
                "background-color": "rgba(37, 99, 235, 0.12)",
                "color": "#2563EB",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown("**Analysis Settings**")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help="Predictions below this level are flagged for manual review.",
    )

    st.markdown("---")
    dot_color = "#22C55E" if model_exists else "#EF4444"
    status_label = "Model Ready" if model_exists else "Model Missing"
    st.markdown(
        f"""
        <div class="status-line">
            <span class="status-dot" style="background:{dot_color};"></span>{status_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header (all pages) — plain text, no box
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 class="page-title">Brain Tumor AI</h1>
    <p class="page-subtitle">
        MRI classification with Grad-CAM explainability and clinical PDF reporting.
        Research and portfolio use only.
    </p>
    """,
    unsafe_allow_html=True,
)

if not model_exists:
    st.error(
        "**Model weights not found.** Train the model first:\n\n"
        "`python train.py --data_dir dataset/ --epochs 30 --fine_tune`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
if selected == "Dashboard":
    render_section("System Overview", "fa-gauge-high")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Model", "EfficientNetB0", "Transfer learning backbone", "fa-network-wired")
    with c2:
        acc = validation_metrics.get("accuracy")
        render_metric_card(
            "Val. Accuracy",
            f"{acc * 100:.1f}%" if acc else "N/A",
            "Hold-out validation set",
            "fa-bullseye",
            "#22C55E" if acc and acc > 0.85 else "#2563EB",
        )
    with c3:
        auc_val = None
        if METRICS_PATH.exists():
            with open(METRICS_PATH, encoding="utf-8") as f:
                auc_val = json.load(f).get("auc_roc")
        render_metric_card(
            "AUC-ROC",
            f"{auc_val:.3f}" if auc_val else "N/A",
            "Discrimination metric",
            "fa-wave-square",
        )
    with c4:
        render_metric_card("AI Version", f"v{AI_SYSTEM_VERSION}", "Report & inference engine", "fa-code-branch")

    render_section("Validation Metrics", "fa-chart-bar")
    if validation_metrics:
        m1, m2, m3, m4 = st.columns(4)
        cards = [
            ("Accuracy", validation_metrics.get("accuracy"), "fa-check-double"),
            ("Precision", validation_metrics.get("precision"), "fa-crosshairs"),
            ("Recall", validation_metrics.get("recall"), "fa-rotate-left"),
            ("F1 Score", validation_metrics.get("f1_score"), "fa-scale-balanced"),
        ]
        for col, (label, val, icon) in zip([m1, m2, m3, m4], cards):
            with col:
                render_metric_card(
                    label,
                    f"{val * 100:.1f}%" if val else "N/A",
                    icon=icon,
                )
    else:
        st.info("Run `python train.py` to generate validation metrics.")

    render_section("Training Artifacts", "fa-folder-open")
    art1, art2 = st.columns(2)
    with art1:
        plot_path = RESULTS_DIR / "accuracy_plot.png"
        if plot_path.exists():
            st.image(str(plot_path), caption="Training curves", use_container_width=True)
    with art2:
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix", use_container_width=True)

# ---------------------------------------------------------------------------
# MRI Analysis
# ---------------------------------------------------------------------------
elif selected == "MRI Analysis":
    render_section("Upload MRI Scan", "fa-cloud-arrow-up")

    uploaded_file = st.file_uploader(
        "Select a brain MRI image (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-state">
                <i class="fa-solid fa-file-medical" style="font-size:2rem; color:#94A3B8;"></i>
                <div class="empty-title">Awaiting MRI upload</div>
                <div class="empty-sub">JPEG or PNG · 224×224 or higher recommended</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_section("Patient Information", "fa-user-injured")
        patient_name = st.text_input(
            "Patient Name",
            value=st.session_state.patient_name,
            placeholder="Enter patient full name (required for clinical report)",
            help="This name will appear on the generated PDF clinical report.",
        )
        st.session_state.patient_name = patient_name.strip()

        if st.button("Run AI Analysis", type="primary", use_container_width=True):
            with st.spinner("Running inference and generating Grad-CAM..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                result = predict(
                    image_path=tmp_path,
                    save_gradcam=True,
                    gradcam_save_path=str(RESULTS_DIR / "gradcam_live.png"),
                    cfg=cfg,
                )
                os.unlink(tmp_path)
                st.session_state.prediction_result = result
                st.session_state.scan_filename = uploaded_file.name
                st.session_state.report_bytes = None
                st.session_state.report_name = None

        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            label = result["prediction"]
            confidence = result["confidence"]
            all_probs = result["all_probs"]
            is_tumor = label == "Tumor"
            result_title = "Tumor Detected" if is_tumor else "No Tumor Detected"
            title_cls = "tumor" if is_tumor else "clear"
            uncertain = confidence < confidence_threshold
            risk_label, _ = assess_risk(confidence)

            render_section("Classification Result", "fa-stethoscope")
            uncertain_html = (
                '<div class="result-note">'
                '<i class="fa-solid fa-triangle-exclamation"></i> '
                "Below confidence threshold — manual review recommended</div>"
                if uncertain else ""
            )
            st.markdown(
                f"""
                <div class="result-block">
                    <div class="result-eyebrow">Analysis Result</div>
                    <div class="result-title {title_cls}">{result_title}</div>
                    <div class="result-meta">
                        Confidence {confidence:.1%} · {risk_label}
                    </div>
                    {uncertain_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            class_names = load_class_names(cfg=cfg)
            render_section("Class Probabilities", "fa-chart-pie")
            for cname, prob in zip(class_names, all_probs):
                disp = "Tumor" if cname.lower() in ("yes", "1", "tumor") else "No Tumor"
                bar_color = "#EF4444" if disp == "Tumor" else "#22C55E"
                st.markdown(
                    f"""
                    <div class="prob-row">
                        <div class="prob-label">{disp}</div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{prob*100:.1f}%; background:{bar_color};"></div>
                        </div>
                        <div class="prob-pct" style="color:{bar_color};">{prob:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_section("Grad-CAM Explainability", "fa-fire")
            col1, col2, col3 = st.columns(3)
            with col1:
                render_scan_image(result["original_img"], "Original MRI")
            with col2:
                render_scan_image(result["heatmap_rgb"], "Grad-CAM Heatmap")
            with col3:
                render_scan_image(result["overlay"], "Attention Overlay")

            render_section("Clinical Report", "fa-file-medical")
            st.markdown(
                '<p class="plain-note">Generate a minimal PDF report with classification, '
                "Grad-CAM images, and validation metrics.</p>",
                unsafe_allow_html=True,
            )

            if st.button("Generate Clinical Report", type="primary", use_container_width=True):
                if not st.session_state.patient_name:
                    st.error("Please enter a patient name before generating the clinical report.")
                else:
                    with st.spinner("Building clinical PDF report..."):
                        pdf_bytes, filepath = build_report_from_prediction(
                            result=result,
                            scan_filename=st.session_state.scan_filename or "scan.jpg",
                            patient_name=st.session_state.patient_name,
                            metrics_path=str(METRICS_PATH),
                            reports_dir=str(REPORTS_DIR),
                        )
                        st.session_state.report_bytes = pdf_bytes
                        st.session_state.report_name = Path(filepath).name
                    st.success(
                        f"Clinical report for **{st.session_state.patient_name}** "
                        f"saved to `{filepath}`"
                    )

            if st.session_state.report_bytes and st.session_state.report_name:
                st.download_button(
                    label="Download Clinical Report",
                    data=st.session_state.report_bytes,
                    file_name=st.session_state.report_name,
                    mime="application/pdf",
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Reports archive
# ---------------------------------------------------------------------------
elif selected == "Reports":
    render_section("Generated Reports", "fa-folder-tree")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(REPORTS_DIR.glob("report_*.pdf"), reverse=True)

    if not reports:
        st.info("No reports generated yet. Run an MRI analysis and click **Generate Clinical Report**.")
    else:
        st.markdown(
            f'<p class="plain-note"><strong>{len(reports)}</strong> report(s) in '
            f"<code>reports/</code></p>",
            unsafe_allow_html=True,
        )
        for rp in reports[:10]:
            with open(rp, "rb") as f:
                st.download_button(
                    label=f"Download {rp.name}",
                    data=f.read(),
                    file_name=rp.name,
                    mime="application/pdf",
                    key=f"dl_{rp.name}",
                )

# ---------------------------------------------------------------------------
# Footer disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-text">
        <strong>Research prototype — not for clinical use.</strong>
        Do not use for medical diagnosis or treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
