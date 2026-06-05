# =============================================================================
# Student Performance Prediction and Academic Risk Analysis System
# Author  : Student ML Project
# Framework: Streamlit  |  Model: Linear Regression (joblib)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CUSTOM CSS
# Forces light background + dark text on EVERY element so the app looks
# identical regardless of whether the user has Streamlit dark-mode enabled.
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject custom CSS – forces white background + dark text everywhere."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        /* ══ RESET – force light theme unconditionally ══════════════════════ */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main, .block-container,
        section[data-testid="stMain"],
        div[data-testid="stVerticalBlock"] {
            background-color: #f1f5f9 !important;
            color: #1e293b !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Every generic text element – force dark */
        p, span, div, li, label, small,
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Streamlit markdown wrapper */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #1e293b !important;
        }

        /* Streamlit widget labels */
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stTextArea label,
        .stRadio label,
        .stCheckbox label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        /* Input boxes */
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox select,
        .stTextArea textarea {
            background: #ffffff !important;
            color: #1e293b !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }
        .stSelectbox [data-baseweb="select"] {
            background: #ffffff !important;
            color: #1e293b !important;
        }

        /* ══ SIDEBAR ════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
            border-right: 1px solid #334155 !important;
        }
        /* Override dark-text reset inside sidebar so it stays white */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #e2e8f0 !important;
        }

        /* ══ HERO BANNER ═══════════════════════════════════════════════════ */
        .hero-banner {
            background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#06b6d4 100%);
            border-radius: 18px;
            padding: 38px 36px;
            margin-bottom: 28px;
        }
        .hero-banner h1 {
            font-size: 2rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            margin: 0 0 8px !important;
        }
        .hero-banner p {
            font-size: 1.05rem !important;
            color: #e0e7ff !important;
            margin: 0 !important;
        }

        /* ══ SECTION HEADING ════════════════════════════════════════════════ */
        .section-heading {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            color: #1e293b !important;
            border-left: 4px solid #6366f1;
            padding-left: 12px;
            margin: 28px 0 16px;
        }

        /* ══ KPI CARD ═══════════════════════════════════════════════════════ */
        .kpi-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 22px 18px;
            text-align: center;
            box-shadow: 0 2px 14px rgba(0,0,0,.09);
            border-top: 4px solid #6366f1;
            transition: transform .2s;
        }
        .kpi-card:hover { transform: translateY(-3px); }
        .kpi-title {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: .07em;
            color: #64748b !important;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 2rem !important;
            font-weight: 800 !important;
            color: #1e293b !important;
        }
        .kpi-sub {
            font-size: 0.78rem !important;
            color: #94a3b8 !important;
            margin-top: 4px;
        }

        /* ══ INFO CARD ══════════════════════════════════════════════════════ */
        .info-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,.06);
            border-top: 3px solid #06b6d4;
            margin-bottom: 14px;
        }
        .info-card strong { color: #1e293b !important; }
        .info-card p, .info-card span, .info-card li {
            color: #374151 !important;
            font-size: .88rem !important;
        }

        /* ══ RESULT CARD ════════════════════════════════════════════════════ */
        .result-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 26px;
            box-shadow: 0 4px 20px rgba(0,0,0,.1);
            border-left: 6px solid #6366f1;
            margin-top: 20px;
        }
        .result-card h3 { color: #1e293b !important; font-size: 1.2rem !important; margin: 0 0 12px !important; }
        .result-card p  { color: #64748b !important; font-size: .88rem !important; margin: 0 0 12px !important; }

        /* ══ RECOMMENDATION BOX ═════════════════════════════════════════════ */
        .rec-box {
            background: linear-gradient(135deg,#eff6ff,#f0fdf4);
            border-radius: 12px;
            padding: 18px 22px;
            border: 1px solid #bfdbfe;
            margin-top: 18px;
        }
        .rec-box h4 { color: #1d4ed8 !important; font-size: .95rem !important; font-weight: 700 !important; margin: 0 0 8px !important; }
        .rec-box p  { color: #1e293b !important; font-size: .88rem !important; margin: 0 !important; line-height: 1.7 !important; }

        /* ══ STATUS BADGES ══════════════════════════════════════════════════ */
        .badge-pass { background:#dcfce7 !important; color:#15803d !important; border-radius:999px; padding:5px 16px; font-weight:700; font-size:.85rem; }
        .badge-fail { background:#fee2e2 !important; color:#dc2626 !important; border-radius:999px; padding:5px 16px; font-weight:700; font-size:.85rem; }
        .badge-safe { background:#dbeafe !important; color:#1d4ed8 !important; border-radius:999px; padding:5px 16px; font-weight:700; font-size:.85rem; }
        .badge-risk { background:#fef3c7 !important; color:#b45309 !important; border-radius:999px; padding:5px 16px; font-weight:700; font-size:.85rem; }

        /* ══ BUTTON ══════════════════════════════════════════════════════════ */
        .stButton > button {
            background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 28px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            transition: all .2s !important;
        }
        .stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

        /* ══ DATAFRAME ═══════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            background: #ffffff !important;
            border-radius: 10px !important;
        }

        /* ══ EXPANDER ════════════════════════════════════════════════════════ */
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }
        [data-testid="stExpander"] summary span {
            color: #1e293b !important;
            font-weight: 600 !important;
        }

        /* ══ DIVIDER ══════════════════════════════════════════════════════════ */
        hr.styled { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }

        /* ══ SUCCESS / ERROR ALERTS ══════════════════════════════════════════ */
        [data-testid="stAlert"] p { color: inherit !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Linear Regression model from disk."""
    model_path = "student_model.joblib"
    if not os.path.exists(model_path):
        st.error(
            "⚠️ **Model file not found!**  "
            "Place `student_model.joblib` in the same folder as `app.py` and restart."
        )
        st.stop()
    return joblib.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (real CSV or synthetic fallback)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Return the Student Performance dataset (CSV or synthetic)."""
    csv_path = "Student_Performance.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["Extracurricular Activities"] = (
            df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
        )
        return df

    rng = np.random.default_rng(42)
    n = 1000
    hours    = rng.integers(1, 10, n).astype(float)
    previous = rng.integers(40, 100, n).astype(float)
    extra    = rng.integers(0, 2, n).astype(float)
    sleep    = rng.integers(4, 10, n).astype(float)
    papers   = rng.integers(0, 9, n).astype(float)
    perf     = np.clip(
        2.5*hours + 0.5*previous + extra + 0.8*sleep + 0.6*papers + rng.normal(0, 4, n),
        0, 100
    )
    return pd.DataFrame({
        "Hours Studied": hours,
        "Previous Scores": previous,
        "Extracurricular Activities": extra,
        "Sleep Hours": sleep,
        "Sample Question Papers Practiced": papers,
        "Performance Index": np.round(perf, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "B-"
    if score >= 60: return "C+"
    if score >= 50: return "C"
    return "F"


def get_gpa(score: float) -> float:
    if score >= 85: return 4.0
    if score >= 80: return 3.7
    if score >= 75: return 3.3
    if score >= 70: return 3.0
    if score >= 65: return 2.7
    if score >= 60: return 2.3
    if score >= 50: return 2.0
    return 0.0


def get_recommendation(score: float, hours: float, sleep: float, papers: float) -> str:
    if score >= 85:
        return ("🌟 Outstanding performance predicted! Maintain your study routine and "
                "consider mentoring peers. You are on track for top academic honours.")
    if score >= 70:
        return ("👍 Good performance predicted! A small push can elevate you to an A. "
                + ("Practise more sample papers. " if papers < 3 else "")
                + ("Ensure 7–8 hours of sleep. " if sleep < 7 else "")
                + "Stay consistent with daily revision.")
    if score >= 50:
        return ("⚠️ Moderate performance predicted. You are passing, but improvement is needed. "
                + ("Increase daily study hours. " if hours < 5 else "")
                + ("Practise more past papers. " if papers < 3 else "")
                + "Seek help from your instructor for weak topics.")
    return ("🚨 At-Risk alert! Immediate academic intervention is recommended. "
            + ("Study hours are very low — aim for 5–6 hours daily. " if hours < 4 else "")
            + ("Poor sleep is hurting performance — target 7–9 hours. " if sleep < 6 else "")
            + "Please contact your academic advisor as soon as possible.")


def kpi_html(title: str, value: str, sub: str = "") -> str:
    """Return HTML for a KPI card with fully explicit inline colours."""
    return (
        f'<div style="background:#ffffff;border-radius:14px;padding:22px 18px;'
        f'text-align:center;box-shadow:0 2px 14px rgba(0,0,0,.09);'
        f'border-top:4px solid #6366f1;">'
        f'<div style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:.07em;color:#64748b;margin-bottom:6px;">{title}</div>'
        f'<div style="font-size:2rem;font-weight:800;color:#1e293b;">{value}</div>'
        f'<div style="font-size:.78rem;color:#94a3b8;margin-top:4px;">{sub}</div>'
        f'</div>'
    )


def section_heading(text: str) -> None:
    """Render a styled section heading."""
    st.markdown(
        f'<div style="font-size:1.2rem;font-weight:700;color:#1e293b;'
        f'border-left:4px solid #6366f1;padding-left:12px;margin:28px 0 16px;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def add_to_history(record: dict) -> None:
    st.session_state.history.insert(0, record)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ① – HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home() -> None:
    st.markdown(
        '<div class="hero-banner">'
        '<h1>🎓 Student Performance Prediction</h1>'
        '<p>AI-powered Academic Risk Analysis System — Linear Regression Model</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    df = get_dataset()
    avg_pi  = df["Performance Index"].mean()
    pass_rt = (df["Performance Index"] >= 50).mean() * 100
    top_pi  = (df["Performance Index"] >= 85).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Records",   f"{len(df):,}",       "in dataset"),         unsafe_allow_html=True)
    c2.markdown(kpi_html("Avg Performance", f"{avg_pi:.1f}",      "Performance Index"),  unsafe_allow_html=True)
    c3.markdown(kpi_html("Pass Rate",       f"{pass_rt:.1f}%",    "Score ≥ 50"),         unsafe_allow_html=True)
    c4.markdown(kpi_html("Top Performers",  f"{top_pi:.1f}%",     "Score ≥ 85"),         unsafe_allow_html=True)

    st.markdown('<hr class="styled">', unsafe_allow_html=True)

    section_heading("📌 System Features")
    col_a, col_b = st.columns(2)
    features = [
        ("🔮 Smart Prediction",    "Predict performance using a trained Linear Regression model."),
        ("📊 Analytics Dashboard", "Visualise trends, distributions, and feature correlations."),
        ("🏅 Grade & GPA",         "Automatic grade letter and GPA calculation."),
        ("⚠️ Risk Detection",      "Identify at-risk students before exams."),
        ("💡 Recommendations",    "Personalised guidance for each student."),
        ("📥 CSV Export",          "Download individual and bulk prediction reports."),
    ]
    for i, (title, desc) in enumerate(features):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(
            f'<div style="background:#ffffff;border-radius:12px;padding:18px 20px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:3px solid #06b6d4;margin-bottom:14px;">'
            f'<span style="font-size:.95rem;font-weight:700;color:#1e293b;">{title}</span><br>'
            f'<span style="color:#475569;font-size:.87rem;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    section_heading("📋 Input Features")
    feature_df = pd.DataFrame({
        "Feature":     ["Hours Studied","Previous Scores","Extracurricular Activities","Sleep Hours","Sample Question Papers Practiced"],
        "Type":        ["Numeric","Numeric","Categorical","Numeric","Numeric"],
        "Range":       ["1 – 9","40 – 100","Yes / No","4 – 10","0 – 9"],
        "Description": [
            "Daily hours spent studying",
            "Marks scored in previous exam",
            "Participation in extracurricular activities",
            "Hours of sleep per night",
            "Number of past papers practised",
        ],
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ② – ABOUT PROJECT
# ─────────────────────────────────────────────────────────────────────────────
def page_about() -> None:
    st.markdown(
        '<div class="hero-banner">'
        '<h1>📖 About This Project</h1>'
        '<p>Research background, methodology, and model details</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    section_heading("🎯 Project Objective")
    st.markdown(
        '<p style="color:#374151;font-size:.95rem;line-height:1.8;">'
        'This system predicts a student\'s <strong>Performance Index</strong> (0–100) using '
        'five behavioural and academic features. It then derives the student\'s grade, GPA, '
        'pass/fail status, and academic risk level, and offers personalised recommendations '
        'to support better learning outcomes.'
        '</p>',
        unsafe_allow_html=True,
    )

    section_heading("🤖 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="background:#ffffff;border-radius:12px;padding:20px;'
            'box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:3px solid #6366f1;">'
            '<p style="color:#374151;font-size:.9rem;line-height:2;margin:0;">'
            '<strong style="color:#1e293b;">Algorithm:</strong> Linear Regression<br>'
            '<strong style="color:#1e293b;">Library:</strong> scikit-learn<br>'
            '<strong style="color:#1e293b;">Target:</strong> Performance Index (continuous)<br>'
            '<strong style="color:#1e293b;">Task Type:</strong> Supervised Regression'
            '</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div style="background:#ffffff;border-radius:12px;padding:20px;'
            'box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:3px solid #06b6d4;">'
            '<p style="color:#374151;font-size:.9rem;line-height:2;margin:0;">'
            '<strong style="color:#1e293b;">Dataset:</strong> Student Performance (10,000 rows)<br>'
            '<strong style="color:#1e293b;">Train / Test Split:</strong> 80% / 20%<br>'
            '<strong style="color:#1e293b;">Saved Format:</strong> Joblib (.joblib)<br>'
            '<strong style="color:#1e293b;">Scaler:</strong> None (features are comparable)'
            '</p></div>',
            unsafe_allow_html=True,
        )

    section_heading("🏆 Grading System")
    grade_df = pd.DataFrame({
        "Grade":       ["A","A-","B+","B","B-","C+","C","F"],
        "Score Range": ["85–100","80–84","75–79","70–74","65–69","60–64","50–59","Below 50"],
        "GPA":         ["4.0","3.7","3.3","3.0","2.7","2.3","2.0","0.0"],
        "Status":      ["Pass","Pass","Pass","Pass","Pass","Pass","Pass","Fail"],
    })
    st.dataframe(grade_df, use_container_width=True, hide_index=True)

    section_heading("🛠️ Technology Stack")
    tech_row1 = st.columns(4)
    tech_row2 = st.columns(4)
    techs1 = ["Python 3.10+", "Streamlit", "scikit-learn", "Pandas / NumPy"]
    techs2 = ["Matplotlib",   "Seaborn",   "Plotly",       "Joblib"]
    for col, tech in zip(tech_row1, techs1):
        col.markdown(
            f'<div style="background:#ffffff;border-radius:12px;padding:16px;text-align:center;'
            f'box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:3px solid #6366f1;margin-bottom:10px;">'
            f'<span style="font-size:.95rem;font-weight:700;color:#1e293b;">⚙️ {tech}</span></div>',
            unsafe_allow_html=True,
        )
    for col, tech in zip(tech_row2, techs2):
        col.markdown(
            f'<div style="background:#ffffff;border-radius:12px;padding:16px;text-align:center;'
            f'box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:3px solid #06b6d4;">'
            f'<span style="font-size:.95rem;font-weight:700;color:#1e293b;">📦 {tech}</span></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ③ – PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction(model) -> None:
    st.markdown(
        '<div class="hero-banner">'
        '<h1>🔮 Student Performance Prediction</h1>'
        '<p>Fill in the student details below to predict academic performance</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        section_heading("👤 Student Information")
        r1c1, r1c2, r1c3 = st.columns(3)
        student_name = r1c1.text_input("Student Name",  placeholder="e.g. Ali Hassan")
        roll_number  = r1c2.text_input("Roll Number",   placeholder="e.g. CS-2024-001")
        subject_name = r1c3.text_input("Subject Name",  placeholder="e.g. Mathematics")

        section_heading("📚 Academic Inputs")
        r2c1, r2c2, r2c3 = st.columns(3)
        hours_studied   = r2c1.number_input("Hours Studied (per day)",      min_value=0.0, max_value=24.0,  value=5.0,  step=0.5)
        previous_scores = r2c2.number_input("Previous Scores (out of 100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        sleep_hours     = r2c3.number_input("Sleep Hours (per night)",       min_value=0.0, max_value=24.0,  value=7.0,  step=0.5)

        r3c1, r3c2 = st.columns(2)
        extra_curr  = r3c1.selectbox("Extracurricular Activities", ["Yes", "No"])
        papers_prac = r3c2.number_input("Sample Question Papers Practiced", min_value=0, max_value=20, value=2, step=1)

        submitted = st.form_submit_button("🚀 Predict Performance", use_container_width=True)

    # ── Validation ────────────────────────────────────────────────────────────
    if submitted:
        errors = []
        if not student_name.strip():   errors.append("Student Name cannot be empty.")
        if not roll_number.strip():    errors.append("Roll Number cannot be empty.")
        if not subject_name.strip():   errors.append("Subject Name cannot be empty.")
        if hours_studied <= 0:         errors.append("Hours Studied must be greater than 0.")
        if not (0 <= previous_scores <= 100):
            errors.append("Previous Scores must be between 0 and 100.")
        if not (0 < sleep_hours <= 24):
            errors.append("Sleep Hours must be between 1 and 24.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            return

        # ── Feature vector ────────────────────────────────────────────────────
        extra_encoded = 1 if extra_curr == "Yes" else 0
        input_df = pd.DataFrame({
            "Hours Studied":                    [hours_studied],
            "Previous Scores":                  [previous_scores],
            "Extracurricular Activities":        [extra_encoded],
            "Sleep Hours":                      [sleep_hours],
            "Sample Question Papers Practiced": [papers_prac],
        })

        # ── Predict ───────────────────────────────────────────────────────────
        raw_score      = float(model.predict(input_df)[0])
        pred_score     = float(np.clip(raw_score, 0, 100))
        pred_pct       = round(pred_score, 2)
        pred_grade     = get_grade(pred_score)
        pred_gpa       = get_gpa(pred_score)
        pass_fail      = "Pass"         if pred_score >= 50 else "Fail"
        risk_status    = "Safe Student" if pred_score >= 50 else "At-Risk Student"
        recommendation = get_recommendation(pred_score, hours_studied, sleep_hours, papers_prac)
        timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── KPI cards ─────────────────────────────────────────────────────────
        section_heading("📊 Prediction Results")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi_html("Performance Index", f"{pred_pct}",   "out of 100"),      unsafe_allow_html=True)
        k2.markdown(kpi_html("Grade",             pred_grade,       f"GPA: {pred_gpa}"),unsafe_allow_html=True)
        k3.markdown(kpi_html("GPA",               str(pred_gpa),   "on 4.0 scale"),    unsafe_allow_html=True)
        k4.markdown(kpi_html("Percentage",        f"{pred_pct}%",  "predicted score"), unsafe_allow_html=True)

        # ── Status badges ─────────────────────────────────────────────────────
        pf_cls   = "badge-pass" if pass_fail   == "Pass"         else "badge-fail"
        risk_cls = "badge-safe" if risk_status == "Safe Student" else "badge-risk"

        st.markdown(
            f'<div class="result-card">'
            f'<h3>🎓 {student_name} &nbsp;'
            f'<span style="font-size:.82rem;color:#64748b;">({roll_number})</span></h3>'
            f'<p>Subject: <strong style="color:#1e293b;">{subject_name}</strong>'
            f' &nbsp;|&nbsp; Timestamp: {timestamp}</p>'
            f'<span class="{pf_cls}">{pass_fail}</span>&nbsp;&nbsp;'
            f'<span class="{risk_cls}">{risk_status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Recommendation ────────────────────────────────────────────────────
        st.markdown(
            f'<div class="rec-box">'
            f'<h4>💡 Personalised Recommendation</h4>'
            f'<p>{recommendation}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Gauge chart ───────────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Performance Index", "font": {"size": 17, "color": "#1e293b"}},
            delta={"reference": 50, "increasing": {"color": "#16a34a"}},
            number={"font": {"color": "#1e293b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar":  {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 50],   "color": "#fee2e2"},
                    {"range": [50, 70],  "color": "#fef9c3"},
                    {"range": [70, 85],  "color": "#dbeafe"},
                    {"range": [85, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 4},
                               "thickness": 0.75, "value": 50},
            },
        ))
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="#ffffff",
            font={"color": "#1e293b"},
            margin=dict(t=60, b=20, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── CSV download ──────────────────────────────────────────────────────
        report_df = pd.DataFrame({
            "Student Name":                [student_name],
            "Roll Number":                 [roll_number],
            "Subject":                     [subject_name],
            "Hours Studied":               [hours_studied],
            "Previous Scores":             [previous_scores],
            "Extracurricular Activities":  [extra_curr],
            "Sleep Hours":                 [sleep_hours],
            "Sample Papers Practiced":     [papers_prac],
            "Predicted Performance Index": [pred_pct],
            "Predicted Grade":             [pred_grade],
            "Predicted GPA":               [pred_gpa],
            "Pass/Fail":                   [pass_fail],
            "Academic Risk Status":        [risk_status],
            "Timestamp":                   [timestamp],
        })
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name=f"report_{roll_number}_{datetime.date.today()}.csv",
            mime="text/csv",
        )

        # ── Save to history ───────────────────────────────────────────────────
        add_to_history({
            "Name":        student_name,
            "Roll No.":    roll_number,
            "Subject":     subject_name,
            "Score":       pred_pct,
            "Grade":       pred_grade,
            "GPA":         pred_gpa,
            "Pass/Fail":   pass_fail,
            "Risk Status": risk_status,
            "Timestamp":   timestamp,
        })
        st.success("✅ Prediction saved to history.")

    # ── History table ─────────────────────────────────────────────────────────
    if st.session_state.history:
        section_heading("🕘 Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name=f"history_{datetime.date.today()}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ④ – ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_analytics() -> None:
    st.markdown(
        '<div class="hero-banner">'
        '<h1>📊 Analytics Dashboard</h1>'
        '<p>Explore dataset insights, correlations, and performance distributions</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    df = get_dataset()

    # ── Dataset KPIs ──────────────────────────────────────────────────────────
    section_heading("📈 Dataset Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(kpi_html("Total Students", f"{len(df):,}",                           "records"), unsafe_allow_html=True)
    s2.markdown(kpi_html("Mean Score",     f"{df['Performance Index'].mean():.1f}",  "average"), unsafe_allow_html=True)
    s3.markdown(kpi_html("Median Score",   f"{df['Performance Index'].median():.1f}","median"),  unsafe_allow_html=True)
    s4.markdown(kpi_html("Std Deviation",  f"{df['Performance Index'].std():.1f}",   "σ"),       unsafe_allow_html=True)

    st.markdown('<hr class="styled">', unsafe_allow_html=True)

    # ── Row 1: Histogram + Boxplot ─────────────────────────────────────────────
    section_heading("📉 Performance Distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df, x="Performance Index", nbins=40,
            title="Performance Index Distribution",
            color_discrete_sequence=["#6366f1"],
        )
        fig_hist.add_vline(x=50, line_dash="dash", line_color="#ef4444",
                           annotation_text="Pass Line (50)", annotation_position="top right")
        fig_hist.update_layout(
            height=370, bargap=0.05,
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font={"color": "#1e293b"},
            title_font={"color": "#1e293b"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        extra_label = df["Extracurricular Activities"].map({1: "Yes", 0: "No"})
        fig_box = px.box(
            df, x=extra_label, y="Performance Index",
            title="Score by Extracurricular Activities",
            color=extra_label,
            color_discrete_map={"Yes": "#6366f1", "No": "#94a3b8"},
            labels={"x": "Extracurricular Activities"},
        )
        fig_box.update_layout(
            height=370, showlegend=False,
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font={"color": "#1e293b"},
            title_font={"color": "#1e293b"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Row 2: Correlation Heatmap ────────────────────────────────────────────
    section_heading("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig_heat, ax = plt.subplots(figsize=(10, 5))
    fig_heat.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, linecolor="#e2e8f0", ax=ax,
        annot_kws={"size": 10, "color": "#1e293b"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold",
                 pad=14, color="#1e293b")
    ax.tick_params(colors="#1e293b", labelsize=9)
    plt.setp(ax.get_xticklabels(), color="#1e293b")
    plt.setp(ax.get_yticklabels(), color="#1e293b")
    fig_heat.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)

    # ── Row 3: Feature Importance + Grade Pie ─────────────────────────────────
    section_heading("🏋️ Feature Importance & Grade Distribution")
    c3, c4 = st.columns(2)
    feat_cols = [
        "Hours Studied", "Previous Scores",
        "Extracurricular Activities", "Sleep Hours",
        "Sample Question Papers Practiced",
    ]

    with c3:
        importances = [abs(corr.loc[f, "Performance Index"]) for f in feat_cols]
        fig_bar = px.bar(
            x=importances, y=feat_cols, orientation="h",
            title="Feature Importance (|correlation| with Target)",
            labels={"x": "Absolute Correlation", "y": "Feature"},
            color=importances,
            color_continuous_scale="Purples",
        )
        fig_bar.update_layout(
            height=370, coloraxis_showscale=False,
            yaxis={"autorange": "reversed"},
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font={"color": "#1e293b"},
            title_font={"color": "#1e293b"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c4:
        df_plot = df.copy()
        df_plot["Grade"] = df_plot["Performance Index"].apply(get_grade)
        grade_counts = df_plot["Grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        fig_pie = px.pie(
            grade_counts, names="Grade", values="Count",
            title="Grade Distribution",
            color_discrete_sequence=px.colors.sequential.Purples_r,
        )
        fig_pie.update_layout(
            height=370,
            paper_bgcolor="#ffffff",
            font={"color": "#1e293b"},
            title_font={"color": "#1e293b"},
            legend_font={"color": "#1e293b"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 4: Scatter ─────────────────────────────────────────────────────────
    section_heading("📐 Hours Studied vs Performance Index")
    df_scatter = df.copy()
    df_scatter["Grade"] = df_scatter["Performance Index"].apply(get_grade)
    fig_scat = px.scatter(
        df_scatter.sample(min(600, len(df_scatter)), random_state=42),
        x="Hours Studied", y="Performance Index",
        color="Grade", trendline="ols", opacity=0.7,
        title="Hours Studied vs Performance Index",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_scat.update_layout(
        height=430,
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font={"color": "#1e293b"},
        title_font={"color": "#1e293b"},
        legend_font={"color": "#1e293b"},
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    # ── Row 5: Sleep trend ────────────────────────────────────────────────────
    section_heading("😴 Average Performance by Sleep Hours")
    sleep_grp = (
        df.groupby("Sleep Hours")["Performance Index"]
        .mean().reset_index()
        .rename(columns={"Performance Index": "Avg Performance"})
    )
    fig_line = px.line(
        sleep_grp, x="Sleep Hours", y="Avg Performance",
        markers=True, title="Average Performance Index by Sleep Hours",
        color_discrete_sequence=["#06b6d4"],
    )
    fig_line.update_traces(line_width=3, marker_size=8)
    fig_line.update_layout(
        height=360,
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font={"color": "#1e293b"},
        title_font={"color": "#1e293b"},
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Raw data + descriptive stats expanders ────────────────────────────────
    with st.expander("🗃️  View Raw Dataset (first 100 rows)"):
        display_df = df.head(100).copy()
        display_df["Extracurricular Activities"] = display_df["Extracurricular Activities"].map({1: "Yes", 0: "No"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("📋  Descriptive Statistics"):
        st.dataframe(
            df[feat_cols + ["Performance Index"]].describe().round(2),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑤ – CONTACT / ABOUT DEVELOPER
# ─────────────────────────────────────────────────────────────────────────────
def page_contact() -> None:
    st.markdown(
        '<div class="hero-banner">'
        '<h1>👨‍💻 About the Developer</h1>'
        '<p>Project details, contact information, and acknowledgements</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_dev, col_proj = st.columns(2)

    def info_block(body: str, accent: str = "#6366f1") -> str:
        return (
            f'<div style="background:#ffffff;border-radius:12px;padding:20px 22px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:3px solid {accent};margin-bottom:14px;">'
            f'{body}</div>'
        )

    with col_dev:
        section_heading("🙋 Developer Profile")
        st.markdown(info_block(
            '<h3 style="color:#1e293b;margin:0 0 10px;">Student Developer</h3>'
            '<p style="color:#475569;font-size:.9rem;line-height:2;margin:0;">'
            '📚 <strong style="color:#1e293b;">Degree:</strong> BS Computer Science / Data Science<br>'
            '🏛️ <strong style="color:#1e293b;">University:</strong> Your University Name<br>'
            '📅 <strong style="color:#1e293b;">Year:</strong> Final Year – 2024 / 2025<br>'
            '💼 <strong style="color:#1e293b;">Specialisation:</strong> Machine Learning & Data Analytics'
            '</p>'
        ), unsafe_allow_html=True)

        section_heading("📬 Contact")
        for label, value in [
            ("📧 Email",    "student@university.edu"),
            ("💼 LinkedIn", "linkedin.com/in/your-profile"),
            ("🐙 GitHub",   "github.com/your-username"),
        ]:
            st.markdown(info_block(
                f'<strong style="color:#1e293b;">{label}:</strong>'
                f'<span style="color:#475569;"> {value}</span>'
            , "#06b6d4"), unsafe_allow_html=True)

    with col_proj:
        section_heading("📁 Project Information")
        st.markdown(info_block(
            '<h3 style="color:#1e293b;margin:0 0 10px;">Final Year Project</h3>'
            '<p style="color:#475569;font-size:.9rem;line-height:2;margin:0;">'
            '🎯 <strong style="color:#1e293b;">Title:</strong> Student Performance Prediction<br>'
            '🤖 <strong style="color:#1e293b;">Algorithm:</strong> Linear Regression<br>'
            '📊 <strong style="color:#1e293b;">Dataset:</strong> 10,000 student records<br>'
            '🧰 <strong style="color:#1e293b;">Stack:</strong> Python · Streamlit · scikit-learn · Plotly<br>'
            '📆 <strong style="color:#1e293b;">Year:</strong> 2024–2025 Academic Year'
            '</p>'
        ), unsafe_allow_html=True)

        section_heading("🙏 Acknowledgements")
        st.markdown(info_block(
            '<ul style="color:#475569;font-size:.88rem;margin:0;padding-left:18px;line-height:2.2;">'
            '<li>Supervisor / Faculty Advisor</li>'
            '<li>Department of Computer Science</li>'
            '<li>Kaggle – Student Performance Dataset</li>'
            '<li>Open-source: scikit-learn, Streamlit, Plotly</li>'
            '</ul>'
        ), unsafe_allow_html=True)

    section_heading("💬 Leave Feedback")
    with st.form("feedback_form"):
        fb_name    = st.text_input("Your Name")
        fb_email   = st.text_input("Your Email")
        fb_message = st.text_area("Message", height=120)
        fb_submit  = st.form_submit_button("📨 Send Feedback")
        if fb_submit:
            if fb_name.strip() and fb_message.strip():
                st.success(f"✅ Thank you, **{fb_name}**! Your feedback has been recorded.")
            else:
                st.error("❌ Please enter your name and a message.")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:20px 0 10px;">'
            '<div style="font-size:2.8rem;">🎓</div>'
            '<div style="font-size:1rem;font-weight:700;color:#e2e8f0;line-height:1.5;">'
            'Student Performance<br>Prediction System</div>'
            '<div style="font-size:.72rem;color:#94a3b8;margin-top:4px;">'
            'AI-Powered Academic Analytics</div>'
            '</div>'
            '<hr style="border-color:#334155;margin:10px 0 20px;">',
            unsafe_allow_html=True,
        )

        pages = {
            "🏠  Home":                "Home",
            "📖  About Project":       "About Project",
            "🔮  Prediction":          "Prediction",
            "📊  Analytics Dashboard": "Analytics Dashboard",
            "👨‍💻  Contact / Developer":  "Contact",
        }
        selected = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

        st.markdown(
            '<hr style="border-color:#334155;margin:20px 0 10px;">'
            '<div style="font-size:.73rem;color:#64748b;text-align:center;padding-bottom:10px;">'
            'v1.0.0 &nbsp;|&nbsp; Linear Regression<br>Built with ❤️ using Streamlit</div>',
            unsafe_allow_html=True,
        )
    return pages[selected]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    init_session_state()
    model = load_model()
    page  = build_sidebar()

    if   page == "Home":               page_home()
    elif page == "About Project":      page_about()
    elif page == "Prediction":         page_prediction(model)
    elif page == "Analytics Dashboard":page_analytics()
    elif page == "Contact":            page_contact()


if __name__ == "__main__":
    main()