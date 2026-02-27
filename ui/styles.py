"""All CSS and HTML constants for the Meeting Process Twin UI."""

import streamlit as st

APP_CSS = """
<style>
/* ── Global ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header banner ────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: #ffffff;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(83,120,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    max-width: 600px;
}

/* ── Metric cards row ─────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 0.25rem;
}

/* colour classes */
.metric-card.green  .value { color: #059669; }
.metric-card.blue   .value { color: #2563eb; }
.metric-card.amber  .value { color: #d97706; }
.metric-card.red    .value { color: #dc2626; }
.metric-card.purple .value { color: #7c3aed; }

/* ── Status bar ───────────────────────────────────────── */
.status-bar {
    background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    font-size: 0.88rem;
}
.status-bar .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-bar .dot.live { background: #059669; animation: pulse 1.5s infinite; }
.status-bar .dot.idle { background: #9ca3af; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Section headers ──────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 1.2rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Sidebar polish ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #fafbfc;
}
[data-testid="stSidebar"] hr {
    margin: 0.8rem 0;
    border-color: #e5e7eb;
}

/* ── Evidence panel ───────────────────────────────────── */
.evidence-item {
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.evidence-item strong {
    color: #1e40af;
}

/* ── Governance cards ─────────────────────────────────── */
.gov-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.gov-accepted {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

/* ── Placeholder card ─────────────────────────────────── */
.placeholder-card {
    border: 2px dashed #cbd5e1;
    padding: 3rem 2rem;
    text-align: center;
    border-radius: 16px;
    color: #94a3b8;
    background: #f8fafc;
}
.placeholder-card h3 {
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* ── hide default metric colours ──────────────────────── */
[data-testid="stMetricDelta"] { display: none; }

/* ── Reduce sidebar widget padding ────────────────────── */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSlider {
    margin-bottom: -0.3rem;
}
</style>
"""

HERO_HTML = """
<div class="hero-banner">
    <h1>Meeting Process Twin</h1>
    <p>Transform meeting recordings into BPMN compliance reports.
       Upload a video, paste the agenda, and watch conformance unfold in real time.</p>
</div>
"""


def inject_styles():
    """Inject all application CSS into the Streamlit page."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
