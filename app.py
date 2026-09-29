"""
DIP Lab — License Plate Deskewing & Unblurring
Professional academic interface. All DIP logic unchanged.
"""

import cv2
import numpy as np
import streamlit as st

from utils.image_utils import generate_sample_license_plate, create_degraded_sample_set
from utils.visualization import plot_side_by_side_histograms, compute_metrics, format_matrix_latex
from dip.transformations import rotate_image, scale_image, deskew_image, composite_transformation
from dip.enhancement import histogram_equalization, laplacian_sharpening, unsharp_masking
from dip.filters import mean_filter, gaussian_filter, median_filter
from dip.noise import add_gaussian_noise, add_salt_pepper_noise
from dip.restoration import bilateral_filter, apply_motion_blur, inverse_filter, wiener_filter
from dip.pipeline import run_full_pipeline

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIP Lab — License Plate Restoration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #F0F4F8; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

/* ─ Top bar ─ */
.topbar {
    background: #fff;
    border-bottom: 1px solid #E2E8F0;
    padding: 14px 32px 12px;
    margin-bottom: 0;
}
.topbar-badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.16em;
    color: #2563EB; text-transform: uppercase;
}
.topbar-title { font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 2px 0; }
.topbar-sub   { font-size: 0.75rem; color: #94A3B8; }

/* ─ Nav pills ─ */
.nav-row { background: #fff; border-bottom: 1px solid #E2E8F0; padding: 0 32px; display:flex; gap:4px; }
.nav-pill {
    display: inline-block; padding: 10px 20px; font-size: 0.83rem;
    font-weight: 600; color: #64748B; cursor: pointer;
    border-bottom: 2px solid transparent; text-decoration: none;
    background: none; border-top: none; border-left: none; border-right: none;
    white-space: nowrap;
}
.nav-pill:hover { color: #2563EB; }
.nav-pill.active { color: #2563EB; border-bottom-color: #2563EB; }

/* ─ Hero ─ */
.hero {
    background: #fff; border-bottom: 1px solid #E2E8F0;
    padding: 56px 48px 48px; text-align: center;
}
.hero-eyebrow {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em;
    color: #2563EB; text-transform: uppercase; margin-bottom: 14px;
}
.hero-h1 {
    font-size: 2.8rem; font-weight: 800; color: #0F172A;
    line-height: 1.15; margin-bottom: 16px;
}
.hero-sub {
    font-size: 1.05rem; color: #475569; line-height: 1.65;
    max-width: 560px; margin: 0 auto 0;
}

/* ─ Upload card ─ */
.upload-wrap {
    border: 2px dashed #CBD5E1; border-radius: 16px;
    padding: 36px 32px; text-align: center;
    background: #F8FAFC; transition: border-color .2s;
    margin-bottom: 4px;
}
.upload-wrap:hover { border-color: #2563EB; }
.upload-h { font-size: 1rem; font-weight: 600; color: #1E293B; margin-bottom: 4px; }
.upload-s { font-size: 0.8rem; color: #94A3B8; }

/* ─ Sample chips ─ */
.chip-label {
    text-align: center; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; color: #94A3B8; text-transform: uppercase;
    margin: 20px 0 10px;
}

/* ─ Main workspace ─ */
.workspace { background: #fff; border-bottom: 1px solid #E2E8F0; padding: 28px 32px 24px; }

/* ─ Eyebrow + heading pattern ─ */
.eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.15em; color: #2563EB; text-transform: uppercase; margin-bottom: 3px; }
.heading  { font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-bottom: 4px; }
.subhead  { font-size: 0.85rem; color: #64748B; line-height: 1.55; margin-bottom: 0; }

/* ─ Technique selector tabs ─ */
.tech-tabs { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 16px; }
.tech-tab {
    padding: 8px 16px; border-radius: 8px; border: 1.5px solid #E2E8F0;
    font-size: 0.82rem; font-weight: 600; color: #475569;
    background: #F8FAFC; cursor: pointer; white-space: nowrap;
    transition: all .15s;
}
.tech-tab:hover { border-color: #93C5FD; color: #1D4ED8; background: #EFF6FF; }
.tech-tab.sel   { border-color: #2563EB; color: #1D4ED8; background: #EFF6FF; }

/* ─ Before/After ─ */
.ba-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; }
.ba-side { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; }
.ba-lbl  { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; color: #94A3B8; text-transform: uppercase; text-align: center; margin-bottom: 10px; }
.ba-cap  { font-size: 0.78rem; color: #94A3B8; text-align: center; margin-top: 8px; }

/* ─ Change items ─ */
.changes { margin-top: 16px; display: flex; flex-direction: column; gap: 6px; }
.chg {
    display: flex; align-items: center; gap: 10px;
    background: #fff; border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 10px 14px; font-size: 0.86rem; color: #334155;
}
.chk { color: #16A34A; font-size: 1rem; flex-shrink: 0; }

/* ─ Info card ─ */
.ic {
    background: #fff; border: 1px solid #E2E8F0;
    border-top: 4px solid #2563EB; border-radius: 12px;
    padding: 20px 22px; margin-top: 16px;
}
.ic-title  { font-size: 0.78rem; font-weight: 800; letter-spacing: 0.12em; color: #0F172A; text-transform: uppercase; }
.ic-module { font-size: 0.72rem; color: #2563EB; font-weight: 600; margin: 2px 0 14px; }
.ic-lbl    { font-size: 0.68rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px; }
.ic-val    { font-size: 0.86rem; color: #334155; line-height: 1.55; margin-bottom: 10px; }

/* ─ Pipeline stepper cards ─ */
.step-card {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 12px; text-align: center;
}
.step-n    { font-size: 0.68rem; font-weight: 700; color: #2563EB; letter-spacing: 0.1em; }
.step-name { font-size: 0.88rem; font-weight: 700; color: #0F172A; margin: 3px 0 2px; }
.step-algo { font-size: 0.72rem; color: #94A3B8; }

/* ─ Metric pills ─ */
.mpill {
    display: inline-block; background: #EFF6FF; color: #1D4ED8;
    font-size: 0.72rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; margin: 2px 3px;
}

/* ─ Pres-mode bar ─ */
.presbar {
    background: #FFFBEB; border: 1.5px solid #FCD34D;
    border-radius: 10px; padding: 10px 20px;
    font-size: 0.85rem; color: #92400E; font-weight: 600;
    margin-bottom: 16px;
}

/* ─ Pres summary cards ─ */
.pscard {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 20px; text-align: center;
}
.pslbl { font-size: 0.68rem; font-weight: 700; color: #2563EB; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }
.psval { font-size: 1rem; font-weight: 700; color: #0F172A; }

/* ─ Streamlit widget overrides ─ */
.stButton > button {
    background: #2563EB !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.87rem !important;
    padding: 0.55rem 1.4rem !important; white-space: nowrap !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }

/* Chip buttons (smaller, outlined) */
.chip-btn button {
    background: #F1F5F9 !important; color: #475569 !important;
    border: 1.5px solid #CBD5E1 !important; border-radius: 20px !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    padding: 6px 14px !important;
}
.chip-btn button:hover { border-color: #93C5FD !important; color: #1D4ED8 !important; background: #EFF6FF !important; }

.stRadio > label { font-size: 0.85rem !important; }
hr { border-color: #E2E8F0 !important; margin: 20px 0 !important; }

/* ─ Streamlit default tab fix ─ */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9 !important; border-radius: 10px; gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; font-size: 0.84rem !important;
    padding: 7px 18px !important; background: transparent !important;
    border: none !important; color: #475569 !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #0F172A !important;
    font-weight: 700 !important; box-shadow: 0 1px 4px rgba(0,0,0,.08) !important;
}

.streamlit-expanderHeader {
    font-size: 0.84rem !important; font-weight: 600 !important;
    color: #475569 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state defaults ────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "img_rgb":        None,
    "active_tech":    None,
    "processed_img":  None,
    "pres_mode":      False,
    "view":           "explore",
    "sub_tech":       {},          # {tech_key: sub_variant}
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _ic(title, module, problem, what, project):
    return f"""<div class="ic">
  <div class="ic-title">{title}</div><div class="ic-module">{module}</div>
  <div class="ic-lbl">Problem</div><div class="ic-val">{problem}</div>
  <div class="ic-lbl">What it does</div><div class="ic-val">{what}</div>
  <div class="ic-lbl">In our project</div><div class="ic-val">{project}</div>
</div>"""

def _ba(before, after, cap_b="Original", cap_a="Processed"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ba-lbl'>BEFORE</div>", unsafe_allow_html=True)
        st.image(before, width="stretch")
        st.markdown(f"<div class='ba-cap'>{cap_b}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='ba-lbl'>AFTER</div>", unsafe_allow_html=True)
        st.image(after, width="stretch")
        st.markdown(f"<div class='ba-cap'>{cap_a}</div>", unsafe_allow_html=True)

def _changes(items):
    html = "<div class='changes'>"
    for it in items:
        html += f"<div class='chg'><span class='chk'>✓</span>{it}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ─── TOP BAR ──────────────────────────────────────────────────────────────────
st.markdown("""<div class="topbar">
  <div class="topbar-badge">DIP Lab · Digital Image Processing</div>
  <div class="topbar-title">Automatic License Plate Deskewing &amp; Unblurring</div>
  <div class="topbar-sub">College Mini-Project · Engineering Demonstration</div>
</div>""", unsafe_allow_html=True)

# Nav using Streamlit columns (no text wrapping because labels are short)
nc = st.columns([1, 1, 1.5, 5])
with nc[0]:
    if st.button("Explore", key="nav_exp"):
        st.session_state.view = "explore"
        st.session_state.active_tech = None
        st.session_state.processed_img = None
        st.rerun()
with nc[1]:
    if st.button("Full Pipeline", key="nav_pip"):
        st.session_state.view = "pipeline"
        st.session_state.processed_img = None
        st.rerun()
with nc[2]:
    if st.session_state.pres_mode:
        if st.button("Exit Pres. Mode", key="pres_off"):
            st.session_state.pres_mode = False
            st.rerun()
    else:
        if st.button("Presentation Mode", key="pres_on"):
            st.session_state.pres_mode = True
            st.rerun()

st.markdown("<hr style='margin:0 0 0 0;'>", unsafe_allow_html=True)

# Presentation mode banner
if st.session_state.pres_mode:
    st.markdown("<div class='presbar'>📽 Presentation Mode — simplified view for classroom display</div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LANDING (no image loaded)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.img_rgb is None:
    st.markdown("""<div class="hero">
      <div class="hero-eyebrow">Digital Image Processing Laboratory</div>
      <div class="hero-h1">License Plate<br>Deskewing &amp; Unblurring</div>
      <div class="hero-sub">
        Explore how classical DIP techniques — geometric transformation,
        image enhancement, denoising, and restoration — correct a degraded
        license plate step by step.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    _, uc, _ = st.columns([1, 2, 1])
    with uc:
        st.markdown("""<div class="upload-wrap">
          <div class="upload-h">Upload your own image</div>
          <div class="upload-s">JPG · PNG · JPEG</div>
        </div>""", unsafe_allow_html=True)
        uf = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uf:
            fb = np.asarray(bytearray(uf.read()), dtype=np.uint8)
            st.session_state.img_rgb = cv2.cvtColor(cv2.imdecode(fb, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            st.session_state.processed_img = None
            st.session_state.active_tech = None
            st.rerun()

    st.markdown("<div class='chip-label'>Or try a built-in sample</div>", unsafe_allow_html=True)
    samples = create_degraded_sample_set()
    CHIP = {
        "Clean Sample Plate":                        "Clean",
        "Skewed Plate (Module I)":                   "Skewed",
        "Low Contrast Plate (Module II)":            "Low Contrast",
        "Salt & Pepper Noisy Plate (Module II/III)": "Noisy",
        "Motion Blurred Plate (Module III)":         "Motion Blur",
    }
    cc = st.columns(len(CHIP))
    for col, (k, img) in zip(cc, samples.items()):
        with col:
            st.markdown("<div class='chip-btn'>", unsafe_allow_html=True)
            if st.button(CHIP[k], key=f"chip_{k}", width="stretch"):
                st.session_state.img_rgb = img
                st.session_state.processed_img = None
                st.session_state.active_tech = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

img_rgb = st.session_state.img_rgb

# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE VIEW
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.view == "pipeline":
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    col_img, col_ctrl = st.columns([1, 2])
    with col_img:
        st.markdown("<div class='eyebrow'>Input</div>", unsafe_allow_html=True)
        st.image(img_rgb, width="stretch")
        if st.button("Change Image", key="chg_pipe"):
            st.session_state.img_rgb = None
            st.session_state.processed_img = None
            st.session_state.active_tech = None
            st.rerun()

    with col_ctrl:
        st.markdown("""
        <div class="eyebrow">Full DIP Pipeline</div>
        <div class="heading">End-to-End Processing</div>
        <div class="subhead">All five DIP stages run sequentially. Each stage's output feeds the next.</div>
        """, unsafe_allow_html=True)

        # All pipeline params with defaults (defined OUTSIDE expander)
        p_deskew_angle = -12.0
        p_deskew_shear =  0.12
        p_eq_method    = "Global Equalization"
        p_add_blur     = True
        p_blur_len     = 11
        p_add_noise    = True
        p_noise_kind   = "Salt & Pepper"
        p_wiener_nsr   = 0.010
        p_sharpen_k    = 1.2

        with st.expander("⚙  Advanced Settings"):
            a1, a2 = st.columns(2)
            with a1:
                p_deskew_angle = st.slider("Deskew Angle (°)", -30.0, 30.0, -12.0, key="p_da")
                p_deskew_shear = st.slider("Shear X",          -0.3,  0.3,   0.12, key="p_ds")
                p_eq_method    = st.selectbox("Enhancement", ["Global Equalization","Adaptive CLAHE"], key="p_eq")
            with a2:
                p_add_blur   = st.checkbox("Simulate Motion Blur",  value=True,  key="p_ab")
                p_blur_len   = st.slider("Blur Length (px)",         3, 21, 11,   key="p_bl")
                p_add_noise  = st.checkbox("Simulate Sensor Noise",  value=True,  key="p_an")
                p_noise_kind = st.selectbox("Noise Type", ["Salt & Pepper","Gaussian"], key="p_nk")
                p_wiener_nsr = st.slider("Wiener NSR (K)", 0.001, 0.05, 0.010, format="%.3f", key="p_wn")
                p_sharpen_k  = st.slider("Sharpen Strength", 0.2, 2.5, 1.2, key="p_sk")

        run_btn = st.button("▶  Run Full DIP Pipeline", key="run_pipe")

    if run_btn:
        with st.spinner("Running pipeline …"):
            stages = run_full_pipeline(
                img_rgb, reference_clean=img_rgb,
                deskew_angle=p_deskew_angle, deskew_shear=p_deskew_shear,
                eq_method=p_eq_method, add_noise=p_add_noise, noise_type=p_noise_kind,
                add_blur=p_add_blur, blur_length=p_blur_len,
                wiener_nsr=p_wiener_nsr, sharpen_strength=p_sharpen_k,
            )

        st.markdown("---")
        st.markdown("<div class='eyebrow'>Results</div><div class='heading'>Stage by Stage</div>",
                    unsafe_allow_html=True)

        # Thumbnail stepper
        tc = st.columns(len(stages))
        for i, (col, s) in enumerate(zip(tc, stages)):
            with col:
                short = s["stage_name"].split(":")[-1].strip() if ":" in s["stage_name"] else s["stage_name"]
                st.markdown(f"""<div class="step-card">
                  <div class="step-n">STEP {i:02d}</div>
                  <div class="step-name">{short}</div>
                  <div class="step-algo">{s['concept']}</div>
                </div>""", unsafe_allow_html=True)
                st.image(s["image"], width="stretch")

        # Expandable detail per stage
        st.markdown("---")
        for i, s in enumerate(stages):
            with st.expander(f"Step {i:02d} — {s['stage_name']}", expanded=(i == len(stages)-1)):
                d1, d2 = st.columns(2)
                with d1:
                    st.image(s["image"], width="stretch")
                with d2:
                    st.markdown(f"""<div class="ic-title">{s['stage_name']}</div>
                    <div class="ic-module">{s['module']} · {s['concept']}</div>
                    <div class="ic-val" style="margin-top:8px;">{s['description']}</div>""",
                                unsafe_allow_html=True)
                    if s["psnr"] > 0:
                        st.markdown(f"<span class='mpill'>PSNR {s['psnr']} dB</span>"
                                    f"<span class='mpill'>MSE {s['mse']}</span>",
                                    unsafe_allow_html=True)

        # Final result
        st.markdown("---")
        st.markdown("<div class='eyebrow'>Final Output</div><div class='heading'>Enhanced Image</div>",
                    unsafe_allow_html=True)
        _, fc, _ = st.columns([1, 2, 1])
        with fc:
            st.image(stages[-1]["image"], width="stretch")

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# EXPLORE VIEW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Row 1: Input image (left) + heading (right) ──────────────────────────────
row_l, row_r = st.columns([1, 2])
with row_l:
    st.markdown("<div class='eyebrow'>Input Image</div>", unsafe_allow_html=True)
    st.image(img_rgb, width="stretch")
    if st.button("Change Image", key="chg_main"):
        st.session_state.img_rgb = None
        st.session_state.processed_img = None
        st.session_state.active_tech = None
        st.rerun()
with row_r:
    st.markdown("""<div class="eyebrow">Choose a DIP Technique</div>
    <div class="heading">What would you like to demonstrate?</div>
    <div class="subhead">Select a technique below to see a Before / After comparison immediately.</div>""",
                unsafe_allow_html=True)

# ── Row 2: Technique selector — full page width so all 5 fit side by side ─────
TECHNIQUES = [
    ("01", "Deskewing",   "Affine Transform",       "geo"),
    ("02", "Enhancement", "Histogram Equalization", "enh"),
    ("03", "Denoising",   "Median / Bilateral",     "den"),
    ("04", "Deblurring",  "Wiener Filtering",       "res"),
    ("05", "Sharpening",  "Laplacian Filter",       "sharp"),
]

t_cols = st.columns(len(TECHNIQUES))
for col, (num, name, algo, key) in zip(t_cols, TECHNIQUES):
    with col:
        selected = st.session_state.active_tech == key
        if st.button(name, key=f"tech_{key}", width="stretch"):
            if st.session_state.active_tech != key:
                st.session_state.active_tech = key
                st.session_state.processed_img = None
            st.rerun()
        algo_style = "color:#2563EB;font-weight:700;" if selected else "color:#94A3B8;"
        sel_mark   = "● " if selected else ""
        st.markdown(f"<p style='text-align:center;font-size:0.72rem;margin-top:-4px;{algo_style}'>{sel_mark}{algo}</p>",
                    unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

ACTIVE = st.session_state.active_tech

if ACTIVE is None:
    st.markdown("""<div style='text-align:center;padding:48px 0;'>
      <div style='font-size:1rem;color:#94A3B8;'>↑ Select a DIP technique above to begin</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE: GEO
# ─────────────────────────────────────────────────────────────────────────────
if ACTIVE == "geo":
    st.markdown("""<div class="eyebrow">Module I — Geometric Transformations</div>
    <div class="heading">Deskewing</div>
    <div class="subhead">Affine transformations correct tilt and skew without distorting parallel lines.</div>""",
                unsafe_allow_html=True)

    # Sub-technique (radio) — default stored in session_state
    if not st.session_state.pres_mode:
        sub_opts = ["Affine Deskewing", "Rotation", "Scaling", "Composite"]
        default_sub = st.session_state.sub_tech.get("geo", "Affine Deskewing")
        sub = st.radio("Variant", sub_opts, index=sub_opts.index(default_sub),
                       horizontal=True, label_visibility="collapsed", key="sub_geo")
        st.session_state.sub_tech["geo"] = sub
    else:
        sub = "Affine Deskewing"

    # All params with defaults OUTSIDE expander
    angle = -12.0; shear_x = 0.15; shear_y = 0.0
    expand = True; fx = 1.5; fy = 0.8
    c_scale = 1.1; c_angle = 20.0; c_shear = 0.1; c_tx = 10.0; c_ty = -5.0

    with st.expander("⚙  Advanced Settings", expanded=False):
        if sub == "Affine Deskewing":
            angle   = st.slider("Correction Angle (°)", -45.0, 45.0, -12.0, 0.5, key="g_ang")
            shear_x = st.slider("Horizontal Shear Sx", -0.5, 0.5, 0.15, 0.01, key="g_sx")
            shear_y = st.slider("Vertical Shear Sy",   -0.5, 0.5,  0.0, 0.01, key="g_sy")
        elif sub == "Rotation":
            angle  = st.slider("Rotation Angle (°)", -180.0, 180.0, 15.0, 1.0, key="g_rot")
            expand = st.checkbox("Expand canvas to fit", value=True, key="g_exp")
        elif sub == "Scaling":
            fx = st.slider("Horizontal Scale fx", 0.2, 3.0, 1.5, 0.1, key="g_fx")
            fy = st.slider("Vertical Scale fy",   0.2, 3.0, 0.8, 0.1, key="g_fy")
        else:
            c_scale = st.slider("Scale",       0.5, 2.0, 1.1, 0.05, key="g_cs")
            c_angle = st.slider("Angle (°)", -90.0, 90.0, 20.0, 1.0, key="g_ca")
            c_shear = st.slider("X-Shear",   -0.4,  0.4,  0.1, 0.02, key="g_csh")
            c_tx    = st.slider("Translate X", -50.0, 50.0, 10.0, 2.0, key="g_ctx")
            c_ty    = st.slider("Translate Y", -50.0, 50.0, -5.0, 2.0, key="g_cty")

    if st.button("Apply Technique", key="apply_geo"):
        if sub == "Affine Deskewing":
            out, _ = deskew_image(img_rgb, angle_deg=angle, shear_x=shear_x, shear_y=shear_y)
        elif sub == "Rotation":
            out, _ = rotate_image(img_rgb, angle, expand=expand)
        elif sub == "Scaling":
            out, _ = scale_image(img_rgb, fx=fx, fy=fy)
        else:
            out, _ = composite_transformation(img_rgb, scale=c_scale, angle_deg=c_angle,
                                              shear_x=c_shear, tx=c_tx, ty=c_ty)
        st.session_state.processed_img = out

    if st.session_state.processed_img is not None:
        st.markdown("---")
        _ba(img_rgb, st.session_state.processed_img, "Original image", "Deskewed result")
        _changes(["Geometric distortion corrected", "License plate aligned horizontally",
                  "Parallel lines preserved (affine property)", "Pixels remapped via 2×3 affine matrix"])
        st.markdown(_ic(
            "AFFINE DESKEWING", "Module I — Geometric Transformations",
            "Camera tilt introduces perspective skew, making characters hard to read.",
            "Applies a 2×3 affine matrix combining rotation and shear. Parallel lines stay parallel.",
            "Corrects the tilt of a photographed license plate before further processing."
        ), unsafe_allow_html=True)
        if not st.session_state.pres_mode:
            with st.expander("Mathematical Detail"):
                st.latex(r"\begin{bmatrix}x'\\y'\end{bmatrix} = M_{2\times3}\begin{bmatrix}x\\y\\1\end{bmatrix}")
                st.latex(r"M_{composite} = M_{shear} \cdot M_{rotation}")

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE: ENH
# ─────────────────────────────────────────────────────────────────────────────
elif ACTIVE == "enh":
    st.markdown("""<div class="eyebrow">Module II — Image Enhancement</div>
    <div class="heading">Contrast Enhancement</div>
    <div class="subhead">Histogram equalization spreads pixel intensities to improve visibility under poor lighting.</div>""",
                unsafe_allow_html=True)

    if not st.session_state.pres_mode:
        sub_opts = ["Histogram Equalization", "Sharpening Filters"]
        default_sub = st.session_state.sub_tech.get("enh", "Histogram Equalization")
        sub = st.radio("Variant", sub_opts, index=sub_opts.index(default_sub),
                       horizontal=True, label_visibility="collapsed", key="sub_enh")
        st.session_state.sub_tech["enh"] = sub
    else:
        sub = "Histogram Equalization"

    # Defaults
    eq_method = "Global Equalization"; clip_limit = 3.0; tile_size = 8
    sh_method = "Laplacian Sharpening"; strength = 1.2; ktype = "4-neighbor"
    ksize = 5; weight = 1.5

    with st.expander("⚙  Advanced Settings", expanded=False):
        if sub == "Histogram Equalization":
            eq_method = st.selectbox("Method", ["Global Equalization","Adaptive CLAHE"], key="e_eq")
            if eq_method == "Adaptive CLAHE":
                clip_limit = st.slider("CLAHE Clip Limit", 1.0, 10.0, 3.0, 0.5, key="e_cl")
                tile_size  = st.slider("Grid Tile Size",   4,   16,   8,   2,   key="e_ts")
        else:
            sh_method = st.selectbox("Sharpening Method", ["Laplacian Sharpening","Unsharp Masking"], key="e_sm")
            if sh_method == "Laplacian Sharpening":
                strength = st.slider("Strength", 0.2, 3.0, 1.2, 0.1, key="e_str")
                ktype    = st.selectbox("Kernel", ["4-neighbor","8-neighbor"], key="e_kt")
            else:
                ksize  = st.slider("Blur Kernel Size", 3, 15, 5, 2, key="e_ks")
                weight = st.slider("Mask Weight k",    0.5, 3.0, 1.5, 0.1, key="e_wt")

    if st.button("Apply Technique", key="apply_enh"):
        if sub == "Histogram Equalization":
            out = histogram_equalization(img_rgb, method=eq_method, clip_limit=clip_limit, tile_size=tile_size)
        else:
            if sh_method == "Laplacian Sharpening":
                out, _ = laplacian_sharpening(img_rgb, strength=strength, kernel_type=ktype)
            else:
                out, _ = unsharp_masking(img_rgb, blur_ksize=ksize, weight=weight)
        st.session_state.processed_img = out

    if st.session_state.processed_img is not None:
        st.markdown("---")
        _ba(img_rgb, st.session_state.processed_img, "Low contrast input", "Enhanced output")
        _changes(["Contrast improved", "Pixel intensity distribution spread wider",
                  "Characters more legible", "Dark areas brightened without clipping"])
        st.markdown(_ic(
            "HISTOGRAM EQUALIZATION", "Module II — Image Enhancement",
            "Poor image contrast — pixel intensities bunched in a narrow range.",
            "Redistributes intensity values uniformly using the Cumulative Distribution Function (CDF).",
            "Improves visibility of license plate characters captured under poor lighting."
        ), unsafe_allow_html=True)
        if not st.session_state.pres_mode:
            with st.expander("Intensity Histograms"):
                fig = plot_side_by_side_histograms(img_rgb, st.session_state.processed_img)
                st.pyplot(fig, width="stretch")
            with st.expander("Mathematical Detail"):
                st.latex(r"s_k = T(r_k) = (L-1)\sum_{j=0}^{k}p_r(r_j)")

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE: DEN
# ─────────────────────────────────────────────────────────────────────────────
elif ACTIVE == "den":
    st.markdown("""<div class="eyebrow">Module II / III — Denoising</div>
    <div class="heading">Noise Removal</div>
    <div class="subhead">Order-statistic filters for impulse noise; edge-preserving filters for Gaussian noise.</div>""",
                unsafe_allow_html=True)

    if not st.session_state.pres_mode:
        sub_opts = ["Median Filter", "Bilateral Filter", "Compare All Filters"]
        default_sub = st.session_state.sub_tech.get("den", "Median Filter")
        sub = st.radio("Variant", sub_opts, index=sub_opts.index(default_sub),
                       horizontal=True, label_visibility="collapsed", key="sub_den")
        st.session_state.sub_tech["den"] = sub
    else:
        sub = "Median Filter"

    # Defaults
    noise_amt = 0.06; ksize = 5; sig_c = 75.0; sig_s = 75.0

    with st.expander("⚙  Advanced Settings", expanded=False):
        noise_amt = st.slider("Noise Density",      0.01, 0.20, 0.06, 0.01, key="d_na")
        ksize     = st.slider("Filter Kernel Size", 3,    11,   5,    2,    key="d_ks")
        if sub == "Bilateral Filter":
            sig_c = st.slider("Sigma Color", 10.0, 150.0, 75.0, 5.0, key="d_sc")
            sig_s = st.slider("Sigma Space", 10.0, 150.0, 75.0, 5.0, key="d_ss")

    if st.button("Apply Technique", key="apply_den"):
        noisy_sp = add_salt_pepper_noise(img_rgb, amount=noise_amt)
        if sub == "Median Filter":
            st.session_state._den_noisy = noisy_sp
            st.session_state.processed_img = median_filter(noisy_sp, ksize)
        elif sub == "Bilateral Filter":
            noisy_g = add_gaussian_noise(img_rgb, sigma=15.0)
            st.session_state._den_noisy = noisy_g
            st.session_state.processed_img = bilateral_filter(
                noisy_g, diameter=ksize, sigma_color=sig_c, sigma_space=sig_s)
        else:  # Compare
            st.session_state._den_noisy = noisy_sp
            st.session_state._den_med   = median_filter(noisy_sp, ksize)
            st.session_state._den_mean, _ = mean_filter(noisy_sp, ksize)
            st.session_state._den_gauss, _ = gaussian_filter(noisy_sp, ksize, 1.5)
            st.session_state.processed_img = st.session_state._den_med

    if st.session_state.processed_img is not None:
        nd = getattr(st.session_state, "_den_noisy", img_rgb)
        st.markdown("---")
        if sub == "Compare All Filters":
            c1, c2, c3, c4 = st.columns(4)
            lbl_img_pairs = [
                (c1, "NOISY INPUT",     nd),
                (c2, "MEAN FILTER",     getattr(st.session_state, "_den_mean",  nd)),
                (c3, "GAUSSIAN FILTER", getattr(st.session_state, "_den_gauss", nd)),
                (c4, "MEDIAN FILTER",   getattr(st.session_state, "_den_med",   nd)),
            ]
            for col, lbl, im in lbl_img_pairs:
                with col:
                    st.markdown(f"<div class='ba-lbl'>{lbl}</div>", unsafe_allow_html=True)
                    st.image(im, width="stretch")
        else:
            cap_a = "Denoised — Bilateral" if sub == "Bilateral Filter" else "Denoised — Median"
            _ba(nd, st.session_state.processed_img, "Noisy input", cap_a)

        _changes(["Impulse noise (salt & pepper) removed", "Character edges preserved",
                  "Extreme pixel values (0/255) discarded", "Non-linear order-statistic filter applied"])
        st.markdown(_ic(
            "MEDIAN FILTER", "Module II / III — Image Enhancement & Restoration",
            "Salt-and-pepper noise introduces random black and white pixels.",
            "Replaces each pixel with the median of its neighbourhood. Extreme outliers are discarded, not averaged.",
            "Cleans sensor noise from license plate images without blurring character edges."
        ), unsafe_allow_html=True)
        if not st.session_state.pres_mode:
            with st.expander("Mathematical Detail"):
                st.latex(r"g(x,y) = \text{median}\{f(s,t) \mid (s,t) \in S_{xy}\}")

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE: RES
# ─────────────────────────────────────────────────────────────────────────────
elif ACTIVE == "res":
    st.markdown("""<div class="eyebrow">Module III — Image Restoration</div>
    <div class="heading">Deblurring</div>
    <div class="subhead">Wiener filtering reverses motion blur in the frequency domain using MMSE estimation.</div>""",
                unsafe_allow_html=True)

    if not st.session_state.pres_mode:
        sub_opts = ["Wiener Filter", "Inverse Filter", "Motion Blur Only"]
        default_sub = st.session_state.sub_tech.get("res", "Wiener Filter")
        sub = st.radio("Variant", sub_opts, index=sub_opts.index(default_sub),
                       horizontal=True, label_visibility="collapsed", key="sub_res")
        st.session_state.sub_tech["res"] = sub
    else:
        sub = "Wiener Filter"

    # Defaults
    blur_len = 13; add_n = True; nsr = 0.01; cutoff = 60.0; eps = 1e-3

    with st.expander("⚙  Advanced Settings", expanded=False):
        blur_len = st.slider("Motion Blur Length (px)", 5, 25, 13, 2, key="r_bl")
        if sub == "Wiener Filter":
            add_n = st.checkbox("Add sensor noise", value=True, key="r_an")
            nsr   = st.slider("Noise-to-Signal Ratio K", 0.0001, 0.1, 0.01, 0.001,
                              format="%.4f", key="r_nsr")
        elif sub == "Inverse Filter":
            cutoff = st.slider("LPF Cutoff D0", 10.0, 150.0, 60.0, 5.0, key="r_co")
            eps    = st.slider("Regularisation ε", 1e-4, 1e-1, 1e-3, format="%.4f", key="r_eps")

    if st.button("Apply Technique", key="apply_res"):
        blurred, psf = apply_motion_blur(img_rgb, length=blur_len, angle_deg=0)
        if sub == "Wiener Filter":
            deg = add_gaussian_noise(blurred, sigma=8.0) if add_n else blurred
            out, _ = wiener_filter(deg, psf, nsr=nsr)
            st.session_state._res_noisy = deg
        elif sub == "Inverse Filter":
            deg = add_gaussian_noise(blurred, sigma=2.0)
            out, _ = inverse_filter(deg, psf, cutoff_radius=cutoff, epsilon=eps)
            st.session_state._res_noisy = deg
        else:  # Motion Blur Only
            out = blurred
            st.session_state._res_noisy = blurred
        st.session_state.processed_img = out

    if st.session_state.processed_img is not None:
        dd = getattr(st.session_state, "_res_noisy", img_rgb)
        mse, psnr = compute_metrics(img_rgb, st.session_state.processed_img)
        st.markdown("---")
        cap_a = "Motion Blur Simulation" if sub == "Motion Blur Only" else f"Restored — PSNR {psnr} dB"
        _ba(dd, st.session_state.processed_img, "Degraded input", cap_a)
        _changes(["Motion blur partially reversed", "Frequency domain MMSE estimation applied",
                  "Noise amplification controlled via K parameter",
                  f"Restoration quality: PSNR {psnr} dB"])
        st.markdown(_ic(
            "WIENER FILTERING", "Module III — Image Restoration",
            "Motion blur and sensor noise degrade license plate characters.",
            "MMSE frequency-domain estimation. Noise term K prevents division-by-zero instability.",
            "Attempts to recover license plate character sharpness after camera motion blur."
        ), unsafe_allow_html=True)
        if not st.session_state.pres_mode:
            with st.expander("Mathematical Detail"):
                st.latex(r"g(x,y) = h(x,y)*f(x,y)+n(x,y)")
                st.latex(r"\hat{F}(u,v)=\left[\frac{H^*(u,v)}{|H(u,v)|^2+K}\right]G(u,v)")

# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE: SHARP
# ─────────────────────────────────────────────────────────────────────────────
elif ACTIVE == "sharp":
    st.markdown("""<div class="eyebrow">Module II — Image Enhancement</div>
    <div class="heading">Sharpening</div>
    <div class="subhead">High-frequency spatial filters enhance edges, making characters crisper and more readable.</div>""",
                unsafe_allow_html=True)

    if not st.session_state.pres_mode:
        sub_opts = ["Laplacian Sharpening", "Unsharp Masking"]
        default_sub = st.session_state.sub_tech.get("sharp", "Laplacian Sharpening")
        sub = st.radio("Variant", sub_opts, index=sub_opts.index(default_sub),
                       horizontal=True, label_visibility="collapsed", key="sub_sharp")
        st.session_state.sub_tech["sharp"] = sub
    else:
        sub = "Laplacian Sharpening"

    # Defaults
    strength = 1.2; ktype = "4-neighbor"; ksize = 5; weight = 1.5

    with st.expander("⚙  Advanced Settings", expanded=False):
        if sub == "Laplacian Sharpening":
            strength = st.slider("Sharpen Strength", 0.2, 3.0, 1.2, 0.1, key="s_str")
            ktype    = st.selectbox("Kernel Type", ["4-neighbor","8-neighbor"], key="s_kt")
        else:
            ksize  = st.slider("Blur Kernel Size", 3, 15, 5, 2, key="s_ks")
            weight = st.slider("Mask Weight k",    0.5, 3.0, 1.5, 0.1, key="s_wt")

    if st.button("Apply Technique", key="apply_sharp"):
        if sub == "Laplacian Sharpening":
            out, kern = laplacian_sharpening(img_rgb, strength=strength, kernel_type=ktype)
            st.session_state._sharp_kern = kern
        else:
            out, _ = unsharp_masking(img_rgb, blur_ksize=ksize, weight=weight)
        st.session_state.processed_img = out

    if st.session_state.processed_img is not None:
        st.markdown("---")
        _ba(img_rgb, st.session_state.processed_img, "Original image", "Sharpened output")
        _changes(["High-frequency edges enhanced", "Character strokes crisper",
                  "Laplacian second derivative subtracted from original",
                  "Fine spatial detail accentuated"])
        st.markdown(_ic(
            "LAPLACIAN SHARPENING", "Module II — Image Enhancement",
            "Image appears soft — characters lack visible sharp edges.",
            "Computes the Laplacian (second spatial derivative) and subtracts it from the original to boost detail.",
            "Makes license plate characters sharper and more legible after restoration."
        ), unsafe_allow_html=True)
        if not st.session_state.pres_mode:
            with st.expander("Mathematical Detail"):
                st.latex(r"f_{sharp}(x,y) = f(x,y) - c\,\nabla^2 f(x,y)")
                if hasattr(st.session_state, "_sharp_kern"):
                    st.markdown("**Spatial Kernel:**")
                    st.write(st.session_state._sharp_kern)

# ─────────────────────────────────────────────────────────────────────────────
# PRESENTATION MODE — summary cards after result
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.pres_mode and st.session_state.processed_img is not None and ACTIVE:
    PMAP = {
        "geo":   ("Affine Transformation",  "Corrects geometric distortion",   "Deskewed plate"),
        "enh":   ("Histogram Equalization", "Redistributes pixel intensities", "Higher contrast"),
        "den":   ("Median Filter",          "Removes impulse noise",           "Clean plate"),
        "res":   ("Wiener Filtering",       "Reverses motion blur (MMSE)",     "Restored characters"),
        "sharp": ("Laplacian Sharpening",   "Enhances high-freq edges",        "Crisper strokes"),
    }
    tech_name, why, result = PMAP[ACTIVE]
    st.markdown("<hr>", unsafe_allow_html=True)
    pp = st.columns(3)
    for col, lbl, val in [(pp[0],"WHAT",tech_name),(pp[1],"WHY",why),(pp[2],"RESULT",result)]:
        with col:
            st.markdown(f"""<div class="pscard">
              <div class="pslbl">{lbl}</div>
              <div class="psval">{val}</div>
            </div>""", unsafe_allow_html=True)
