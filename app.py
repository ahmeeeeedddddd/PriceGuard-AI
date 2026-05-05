"""
app.py — PriceGuard AI 6-Zone Monitoring Dashboard
==================================================
Real-time dashboard divided into six monitoring zones:

ZONE 1 Cluster Health     → Centroid stability & Drift signals
ZONE 2 Live Scoring Feed  → 5-signal breakdown of recent cases
ZONE 3 Action Dispatch    → ReAct routing paths & dynamic actions
ZONE 4 SHAP Reliability   → Implicit RAG (cosine) & Feature explanations
ZONE 5 Outcome Tracking   → Alert log & action effectiveness
ZONE 6 System Alerts      → Critical logs, drift fires, API health
"""

import os
import sys
import warnings
import logging
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from PIL import Image
from dotenv import load_dotenv

warnings.simplefilter("ignore", FutureWarning)
load_dotenv()

# ── Ensure priceguard_ai is importable ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PriceGuard AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .zone-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        height: 100%;
    }
    .zone-title {
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #c9d1d9; }
    .metric-label { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; }
    .hero-bg {
        background: linear-gradient(90deg, #1f6feb 0%, #111 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #1f6feb, #38d353); }
    .badge { padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-blue { background: #1f6feb; }
    .badge-red { background: #da3633; }
    .badge-green { background: #238636; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "drift_report" not in st.session_state:
    st.session_state.drift_report = None

# ── Utility ──────────────────────────────────────────────────────────────────
def load_labeled_data():
    path = os.path.join(BASE_DIR, "data", "labeled_products.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-bg">
    <h1 style='margin:0; color:white;'>🛡️ PriceGuard AI : <span style='color:#58a6ff'>Monitoring Dashboard</span></h1>
    <p style='color:#8b949e; margin-top:0.5rem;'>Intelligent Sense-Reason-Act Pipeline | Autonomous Agent Operations</p>
</div>
""", unsafe_allow_html=True)

# ── Dashboard Layout ──────────────────────────────────────────────────────────
z1, z2, z3 = st.columns(3)
z4, z5, z6 = st.columns(3)

# ── ZONE 1: Cluster Health ───────────────────────────────────────────────────
with z1:
    st.markdown('<div class="zone-box"><div class="zone-title">ZONE 1 · Cluster Health</div>', unsafe_allow_html=True)
    df_l = load_labeled_data()
    if not df_l.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-label">Population</div><div class="metric-val">{len(df_l)}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-label">Avg Price</div><div class="metric-val">${df_l.price.mean():.1f}</div>', unsafe_allow_html=True)
        
        fig_pie = px.pie(df_l, names='cluster_label', hole=0.4, 
                         color_discrete_sequence=['#da3633', '#f0883e', '#238636'])
        fig_pie.update_layout(showlegend=False, height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Waiting for data...")
    st.markdown('</div>', unsafe_allow_html=True)

# ── ZONE 2: Live Scoring Feed ────────────────────────────────────────────────
with z2:
    st.markdown('<div class="zone-box"><div class="zone-title">ZONE 2 · Live Scoring Feed</div>', unsafe_allow_html=True)
    if st.session_state.history:
        latest = st.session_state.history[-1]
        score = latest["score"]
        color = "#38d353" if score < 0.5 else ("#f0883e" if score < 0.75 else "#da3633")
        
        st.markdown(f"<div style='text-align:center;'><h2 style='color:{color}; font-size:3rem;'>{score:.2f}</h2><p class='metric-label'>Latest Intelligent Score</p></div>", unsafe_allow_html=True)
        
        # 5-signal breakdown
        signals = latest["score_breakdown"]
        cols = st.columns(5)
        titles = ["S", "Gap", "μ", "ARI", "SHAP"]
        keys = ["s", "gap", "mu", "arima", "shap_cosine"]
        for col, t, k in zip(cols, titles, keys):
            with col:
                st.markdown(f"<div style='text-align:center;'><p style='font-size:0.7rem; color:#8b949e;'>{t}</p><p style='font-weight:700;'>{signals[k]:.1f}</p></div>", unsafe_allow_html=True)
    else:
        st.info("Run inference to see live scoring.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── ZONE 3: Action Dispatch ──────────────────────────────────────────────────
with z3:
    st.markdown('<div class="zone-box"><div class="zone-title">ZONE 3 · Action Dispatch</div>', unsafe_allow_html=True)
    if st.session_state.history:
        latest = st.session_state.history[-1]
        action = latest["action"]
        st.markdown(f"**Path Triggered:** `{action['path_triggered']}`")
        st.markdown(f"**Action Type:** `{action['action_type']}`")
        st.markdown(f"**Timing:** `{action['timing']}`")
        st.markdown(f"**Intensity:** `{action['intensity']}/10`")
        st.info(action["personalisation"])
    else:
        st.info("Actions will appear here.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── ZONE 4: SHAP Reliability ────────────────────────────────────────────────
with z4:
    st.markdown('<div class="zone-box"><div class="zone-title">ZONE 4 · SHAP Reliability</div>', unsafe_allow_html=True)
    if st.session_state.history:
        latest = st.session_state.history[-1]
        cos = latest["shap_cosine"]
        st.write(f"**Cosine Similarity (Implicit RAG):** `{cos:.3f}`")
        st.progress(cos)
        if cos < 0.80:
            st.warning("⚠️ Low reliability - ReAct re-check loop triggered.")
        else:
            st.success("✅ High reasoning reliability")
        
        # Dominant Feature
        st.markdown(f"**Dominant Reasoning Driver:** `{latest['dominant_shap']}`")
    else:
        st.info("SHAP metrics available after run.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── ZONE 5: Outcome Tracking ─────────────────────────────────────────────────
with z5:
    st.markdown('<div class="zone-box"><div class="zone-title">ZONE 5 · Outcome Tracking</div>', unsafe_allow_html=True)
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history).tail(5)
        st.table(df_hist[["product_name", "price", "score", "path_taken"]])
    else:
        st.info("Log is empty.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── ZONE 6: System Alerts ────────────────────────────────────────────────────
with z6:
    st.markdown('<div class="zone-box"><div class="zone-title">ZONE 6 · System Alerts</div>', unsafe_allow_html=True)
    if st.session_state.drift_report:
        dr = st.session_state.drift_report
        s1 = dr["signal_1"]["fired"]
        s2 = dr["signal_2"]["fired"]
        
        st.markdown("**Drift Status:** " + ("🔴 TRIGGERED" if dr["drift_triggered"] else "🟢 STABLE"))
        st.markdown(f"- Signal 1 (EMA Conf): {'❌' if s1 else '✅'} {dr['signal_1']['ema_confidence']:.2f}")
        st.markdown(f"- Signal 2 (Centroid): {'❌' if s2 else '✅'} {dr['signal_2']['centroid_dist']:.2f}")
        
        if dr["drift_triggered"]:
            st.error("AUTONOMOUS RETRAINING IN PROGRESS...")
    else:
        st.success("System monitoring startup successful.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Sidebar Control ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092218.png", width=100)
    st.markdown("### Agent Controls")
    
    with st.form("manual_entry"):
        p_name = st.text_input("Product Name", "ASUS Gaming Laptop")
        p_price = st.number_input("Price", 1200.0)
        p_cat = st.text_input("Category", "Electronics")
        submit = st.form_submit_button("Run ReAct Loop")
        
    if submit:
        from react_loop import run_react_loop
        from drift_monitor import check_drift
        
        product = {
            "product_name": p_name,
            "price": p_price,
            "category": p_cat,
            "rating": 4.5,
            "review_count": 50,
            "discount_percentage": 5.0,
            "stock_status": 1
        }
        
        with st.spinner("Agent Reasoning..."):
            res = run_react_loop(product)
            st.session_state.history.append(res)
            
            # Check drift
            st.session_state.drift_report = check_drift(res["confidence"], product, res["label"])
            st.rerun()

    if st.button("Manual Recluster"):
        from drift_monitor import trigger_outer_loop
        trigger_outer_loop()
        st.success("Outer loop complete.")
