from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quant_engine import (
    DEFAULT_ALMGREN_GAMMA,
    DEFAULT_BROKER_COMM_RATE,
    DEFAULT_MIN_BROKER_COMM_HKD,
    HK_DEFAULT_LOT_SIZES,
    HK_SHORT_ELIGIBLE_WHITELIST,
    is_hk_short_eligible,
    analyze_portfolio_health,
    analyze_stock_trend,
    evaluate_historical_prediction_accuracy,
    fetch_hk_macro_regime,
    generate_day_trading_signal,
    load_portfolio_config,
    normalize_hk_ticker,
    save_portfolio_config,
)

st.set_page_config(
    page_title="香港股市預測 | HKEX AlphaEngine 機構級量化終端",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_terminal_and_portal_css() -> None:
    """
    Inject institutional dark Bento UI CSS + Special-Effects '香港股市預測' Portal animations.
    Ensures zero code blocks are shown and all pages are clean, visual, and easy to understand.
    """
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;700;900&family=JetBrains+Mono:wght@500;600;700&display=swap');

    @keyframes ambientAuroraDrift {
        0%   { transform: translate3d(-6%, -2%, 0) rotate(-3deg) scale(1.0); opacity: 0.48; }
        50%  { transform: translate3d(6%, 3%, 0) rotate(3deg) scale(1.08); opacity: 0.78; }
        100% { transform: translate3d(-6%, -2%, 0) rotate(-3deg) scale(1.0); opacity: 0.48; }
    }

    @keyframes specularBeamMove {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes liveDotPulse {
        0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        50%      { transform: scale(1.18); opacity: 0.85; box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
    }

    @keyframes neonTitleShimmer {
        0%   { background-position: 0% 50%; filter: drop-shadow(0 0 18px rgba(56, 189, 248, 0.45)); }
        50%  { background-position: 100% 50%; filter: drop-shadow(0 0 32px rgba(16, 185, 129, 0.68)); }
        100% { background-position: 0% 50%; filter: drop-shadow(0 0 18px rgba(56, 189, 248, 0.45)); }
    }

    @keyframes cyberGridFloat {
        0%   { background-position: 0px 0px, 0px 0px; }
        100% { background-position: 0px 48px, 48px 0px; }
    }

    @keyframes radarRingPulse {
        0%   { transform: scale(0.85); opacity: 0.75; }
        70%  { transform: scale(1.35); opacity: 0; }
        100% { transform: scale(1.35); opacity: 0; }
    }

    @keyframes floatUpSlow {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-8px); }
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide any accidental code blocks to guarantee a clean, zero-code user experience */
    pre, code {
        display: none !important;
    }

    /* Ambient Top Spotlight */
    .stApp::before {
        content: "";
        position: fixed;
        top: -160px;
        left: 10%;
        width: 80vw;
        height: 340px;
        background: radial-gradient(
            ellipse at center,
            rgba(56, 189, 248, 0.15) 0%,
            rgba(99, 102, 241, 0.10) 42%,
            rgba(16, 185, 129, 0.06) 70%,
            rgba(0, 0, 0, 0) 85%
        );
        filter: blur(60px);
        pointer-events: none;
        z-index: 0;
        animation: ambientAuroraDrift 15s ease-in-out infinite;
    }

    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }

    /* =====================================================
       Special-Effects Portal Hero ("香港股市預測")
       ===================================================== */
    .hk-portal-stage {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 52px 38px 44px 38px;
        margin-bottom: 24px;
        background:
            radial-gradient(circle at 18% 22%, rgba(14, 165, 233, 0.22) 0%, transparent 45%),
            radial-gradient(circle at 82% 75%, rgba(16, 185, 129, 0.20) 0%, transparent 48%),
            radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.14) 0%, transparent 60%),
            linear-gradient(160deg, rgba(9, 14, 28, 0.96) 0%, rgba(15, 23, 42, 0.94) 55%, rgba(6, 11, 22, 0.98) 100%);
        border: 1px solid rgba(56, 189, 248, 0.32);
        box-shadow:
            0 25px 60px -15px rgba(0, 0, 0, 0.75),
            0 0 40px -10px rgba(56, 189, 248, 0.22),
            inset 0 1px 0 rgba(255, 255, 255, 0.22);
        text-align: center;
    }

    /* Animated 3D Cyber Financial Grid Overlay */
    .hk-portal-stage::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(to right, rgba(56, 189, 248, 0.07) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(56, 189, 248, 0.07) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: radial-gradient(ellipse at center, rgba(0,0,0,0.85) 20%, transparent 78%);
        -webkit-mask-image: radial-gradient(ellipse at center, rgba(0,0,0,0.85) 20%, transparent 78%);
        animation: cyberGridFloat 6s linear infinite;
        pointer-events: none;
    }

    /* Specular Top Laser Sweep */
    .hk-portal-stage::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2.5px;
        background: linear-gradient(
            90deg,
            transparent 0%,
            #38BDF8 25%,
            #34D399 50%,
            #A855F7 75%,
            transparent 100%
        );
        background-size: 200% 100%;
        animation: specularBeamMove 5s linear infinite;
    }

    .portal-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.38);
        border-radius: 999px;
        padding: 6px 18px;
        font-size: 0.84rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: #7DD3FC;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
    }

    .portal-main-title {
        font-family: 'Noto Sans TC', 'Inter', sans-serif;
        font-size: clamp(2.8rem, 5.6vw, 4.6rem);
        font-weight: 900;
        letter-spacing: 6px;
        margin: 4px 0 14px 0;
        background: linear-gradient(
            110deg,
            #FFFFFF 0%,
            #7DD3FC 22%,
            #34D399 48%,
            #FDE047 72%,
            #38BDF8 90%,
            #FFFFFF 100%
        );
        background-size: 260% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: neonTitleShimmer 7s ease-in-out infinite;
    }

    .portal-subtitle {
        max-width: 760px;
        margin: 0 auto 28px auto;
        font-size: 1.05rem;
        line-height: 1.65;
        color: #CBD5E1;
        font-weight: 400;
    }

    .portal-orb-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-top: 22px;
        text-align: left;
    }

    .portal-stock-card {
        position: relative;
        background: linear-gradient(155deg, rgba(15, 23, 42, 0.88) 0%, rgba(11, 17, 32, 0.94) 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 18px 20px;
        backdrop-filter: blur(12px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        animation: floatUpSlow 6s ease-in-out infinite;
    }

    .portal-stock-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 16px 36px -8px rgba(14, 165, 233, 0.25);
    }

    /* Top Macro Status Ribbon */
    .macro-ribbon {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, rgba(9, 13, 24, 0.92) 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-top: 1px solid rgba(56, 189, 248, 0.38);
        border-radius: 12px;
        padding: 11px 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .macro-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        font-size: 0.96rem;
        color: #F8FAFC;
        letter-spacing: 0.4px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
        animation: liveDotPulse 2.4s infinite;
    }

    .macro-pills {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        font-size: 0.82rem;
        color: #94A3B8;
    }

    .macro-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .mono-num {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-weight: 600;
        color: #F1F5F9;
    }

    .badge-bull {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 9px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.76rem;
    }

    .badge-bear {
        background: rgba(244, 63, 94, 0.14);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
        padding: 2px 9px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.76rem;
    }

    .badge-warn {
        background: rgba(245, 158, 11, 0.14);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 2px 9px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.76rem;
    }

    /* Bento Grid Cards with Specular Top Light Movement */
    .bento-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(165deg, rgba(17, 24, 39, 0.82) 0%, rgba(11, 15, 25, 0.92) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 8px 26px -8px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .bento-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1.5px;
        background: linear-gradient(
            90deg,
            rgba(56, 189, 248, 0) 0%,
            rgba(56, 189, 248, 0.65) 50%,
            rgba(56, 189, 248, 0) 100%
        );
        background-size: 200% 100%;
        animation: specularBeamMove 8s linear infinite;
    }

    .bento-card:hover {
        border-color: rgba(56, 189, 248, 0.32);
        box-shadow: 0 12px 32px -6px rgba(14, 165, 233, 0.14);
    }

    .bento-title {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .exec-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 12px;
    }

    .exec-cell {
        background: rgba(30, 41, 59, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 10px;
        padding: 10px 12px;
    }

    .exec-label {
        font-size: 0.74rem;
        color: #94A3B8;
        margin-bottom: 3px;
    }

    .exec-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.06rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .factor-row {
        margin-bottom: 11px;
    }

    .factor-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        margin-bottom: 4px;
        color: #CBD5E1;
    }

    .factor-track {
        width: 100%;
        height: 7px;
        background: rgba(30, 41, 59, 0.85);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }

    .factor-fill-pos {
        height: 100%;
        background: linear-gradient(90deg, #0EA5E9, #10B981);
        border-radius: 999px;
    }

    .factor-fill-neg {
        height: 100%;
        background: linear-gradient(90deg, #F59E0B, #F43F5E);
        border-radius: 999px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(17, 24, 39, 0.78), rgba(11, 15, 25, 0.90));
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-top: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 13px 16px;
        backdrop-filter: blur(10px);
        transition: border-color 0.25s ease, transform 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8 !important;
    }

    div[data-testid="stMetricValue"] div {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-size: 1.32rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.3px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


inject_terminal_and_portal_css()


# ==========================================
# Cached Quantitative Data Pipelines
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def cached_macro_regime() -> dict:
    return fetch_hk_macro_regime()


@st.cache_data(ttl=300, show_spinner=False)
def cached_day_trading_signal(
    ticker: str,
    min_adv_hkd: float,
    interval: str,
    broker_comm_rate: float,
    min_broker_comm_hkd: float,
    almgren_gamma: float,
):
    return generate_day_trading_signal(
        ticker,
        min_adv_hkd=min_adv_hkd,
        interval=interval,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_stock_trend(ticker: str):
    return analyze_stock_trend(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_portfolio_health(
    holdings_json: str,
    total_cap: float,
    rf: float,
    max_w: float,
    min_adv: float,
    method: str,
    trials: int,
    cov_est: str,
    max_to: float,
    comm_rate: float,
    min_comm: float,
    gamma: float,
):
    holdings = json.loads(holdings_json)
    return analyze_portfolio_health(
        holdings,
        total_capital=total_cap,
        rf_rate=rf,
        max_single_weight=max_w,
        min_adv_hkd=min_adv,
        alloc_method=method,
        num_trials=trials,
        cov_estimator=cov_est,
        max_turnover=max_to,
        broker_comm_rate=comm_rate,
        min_broker_comm_hkd=min_comm,
        almgren_gamma=gamma,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_historical_accuracy(
    ticker: str,
    horizon: int,
    history_period: str,
    atr_stop_mult: float,
    signal_z_threshold: float,
    comm_rate: float,
    min_comm: float,
    gamma: float,
):
    return evaluate_historical_prediction_accuracy(
        ticker=ticker,
        forward_horizon=horizon,
        history_period=history_period,
        atr_stop_mult=atr_stop_mult,
        signal_z_threshold=signal_z_threshold,
        broker_comm_rate=comm_rate,
        min_broker_comm_hkd=min_comm,
        almgren_gamma=gamma,
    )


def render_factor_bars_html(title: str, subtitle: str, factor_dict: dict, max_scale: float = 3.0, unit: str = "σ") -> str:
    """Render sleek horizontal factor exposure bars inside a Bento card."""
    rows_html = ""
    for label, val in factor_dict.items():
        pct = min(100.0, max(6.0, (abs(val) / max_scale) * 100.0))
        fill_cls = "factor-fill-pos" if val >= 0 else "factor-fill-neg"
        val_color = "#34D399" if val > 0 else ("#FB7185" if val < 0 else "#94A3B8")
        rows_html += f"""
        <div class="factor-row">
            <div class="factor-header">
                <span>{label}</span>
                <span class="mono-num" style="color: {val_color};">{val:+.2f} {unit}</span>
            </div>
            <div class="factor-track">
                <div class="{fill_cls}" style="width: {pct:.1f}%;"></div>
            </div>
        </div>
        """
    return f"""
    <div class="bento-card">
        <div class="bento-title">
            <span>{title}</span>
            <span style="color: #38BDF8; font-size: 0.74rem;">{subtitle}</span>
        </div>
        {rows_html}
    </div>
    """


# ==========================================
# Task 1.5: Reactive State & Atomic JSON Sync
# ==========================================
if "portfolio_config" not in st.session_state:
    st.session_state.portfolio_config = load_portfolio_config()
if "last_saved_hash" not in st.session_state:
    st.session_state.last_saved_hash = json.dumps(st.session_state.portfolio_config, sort_keys=True, ensure_ascii=False)
if "entered_main_terminal" not in st.session_state:
    # Show the fancy "香港股市預測" special-effects portal page first before the main page
    st.session_state.entered_main_terminal = False

cfg = st.session_state.portfolio_config

QUICK_WATCHLIST = {
    "0700.HK": "0700.HK 騰訊控股",
    "0005.HK": "0005.HK 匯豐控股",
    "3690.HK": "3690.HK 美團-W",
    "9988.HK": "9988.HK 阿里巴巴-W",
    "1810.HK": "1810.HK 小米集團-W",
    "0388.HK": "0388.HK 香港交易所",
    "1211.HK": "1211.HK 比亞迪股份",
    "2475.HK": "2475.HK 立訊精密/中小型股",
}

with st.sidebar:
    st.markdown("### 🎛️ 香港股市預測控制中心")
    if st.session_state.entered_main_terminal:
        if st.button("✨ 返回「香港股市預測」特效前導頁", use_container_width=True):
            st.session_state.entered_main_terminal = False
            st.rerun()
    else:
        if st.button("🚀 直接進入量化分析主終端", use_container_width=True, type="primary"):
            st.session_state.entered_main_terminal = True
            st.rerun()

    st.markdown("#### 🎯 單股追蹤標的")
    preset_t = st.selectbox(
        "快速切換熱門港股",
        options=list(QUICK_WATCHLIST.keys()),
        format_func=lambda k: QUICK_WATCHLIST[k],
        index=0,
    )
    custom_t = st.text_input("或輸入任意港股代號 (如 0941.HK)", value=preset_t).strip().upper()
    active_ticker = normalize_hk_ticker(custom_t) if custom_t else preset_t

    interval_choice = st.segmented_control(
        "日內當沖 K 線週期",
        options=["15m", "5m", "1d"],
        default="15m",
    ) or "15m"

    st.markdown("#### 🛡️ 投資組合與矩陣去噪設定")
    method_options = {
        "hrp": "HRP 階層風險平價 (推薦)",
        "risk_parity": "Risk Parity 等風險貢獻",
        "min_variance": "Minimum Variance 最小變異數",
    }
    curr_method = cfg.get("alloc_method", "hrp")
    alloc_method = st.selectbox(
        "資產配置數學模型",
        options=list(method_options.keys()),
        format_func=lambda k: method_options[k],
        index=list(method_options.keys()).index(curr_method) if curr_method in method_options else 0,
    )

    cov_options = {
        "ledoit_wolf": "Ledoit-Wolf 壓縮共變異數",
        "rmt": "RMT 隨機矩陣特徵值去噪",
    }
    curr_cov = cfg.get("cov_estimator", "ledoit_wolf")
    cov_estimator = st.selectbox(
        "共變異數矩陣去噪引擎",
        options=list(cov_options.keys()),
        format_func=lambda k: cov_options[k],
        index=list(cov_options.keys()).index(curr_cov) if curr_cov in cov_options else 0,
    )

    total_capital_hkd = st.number_input(
        "目標配置總資金 (HKD)",
        min_value=10000.0,
        value=float(cfg.get("total_capital_hkd", 300000.0)),
        step=10000.0,
    )
    rf_rate_pct = st.slider(
        "無風險利率 Rf (% 年化)",
        min_value=0.0,
        max_value=8.0,
        value=float(cfg.get("risk_free_rate", 0.04)) * 100.0,
        step=0.25,
    )
    max_pos_pct = st.slider(
        "單一持股權重上限 (%)",
        min_value=15.0,
        max_value=60.0,
        value=float(cfg.get("max_single_position_pct", 0.35)) * 100.0,
        step=5.0,
    )
    max_turnover_pct = st.slider(
        "單次調倉換手率上限 (%)",
        min_value=10.0,
        max_value=100.0,
        value=float(cfg.get("max_turnover", 0.40)) * 100.0,
        step=5.0,
    )

    with st.expander("🏛️ 港交所規費與市場衝擊滑價參數", expanded=False):
        broker_comm_bps = st.number_input(
            "經紀佣金率 (% 費率，預設 0.03%)",
            min_value=0.0,
            max_value=0.50,
            value=float(cfg.get("broker_comm_rate", DEFAULT_BROKER_COMM_RATE)) * 100.0,
            step=0.01,
            format="%.3f",
        )
        min_broker_comm = st.number_input(
            "最低經紀佣金 (HKD，預設 15 元)",
            min_value=0.0,
            max_value=200.0,
            value=float(cfg.get("min_broker_comm_hkd", DEFAULT_MIN_BROKER_COMM_HKD)),
            step=5.0,
        )
        almgren_gamma = st.slider(
            "Almgren-Chriss 衝擊係數 γ",
            min_value=0.10,
            max_value=1.00,
            value=float(cfg.get("almgren_gamma", DEFAULT_ALMGREN_GAMMA)),
            step=0.05,
        )
        min_adv_input = st.number_input(
            "最低 20 日均成交額門檻 (HKD)",
            min_value=0,
            value=int(cfg.get("min_adv_hkd", 5000000)),
            step=1000000,
        )
        num_trials = st.number_input(
            "回測多重檢定次數 (DSR K)",
            min_value=1,
            max_value=500,
            value=int(cfg.get("num_backtest_trials", 5)),
            step=1,
        )

    # Reactive update & automatic atomic persistence whenever parameters change
    cfg["total_capital_hkd"] = float(total_capital_hkd)
    cfg["risk_free_rate"] = float(rf_rate_pct) / 100.0
    cfg["max_single_position_pct"] = float(max_pos_pct) / 100.0
    cfg["max_turnover"] = float(max_turnover_pct) / 100.0
    cfg["min_adv_hkd"] = float(min_adv_input)
    cfg["alloc_method"] = alloc_method
    cfg["cov_estimator"] = cov_estimator
    cfg["broker_comm_rate"] = float(broker_comm_bps) / 100.0
    cfg["min_broker_comm_hkd"] = float(min_broker_comm)
    cfg["almgren_gamma"] = float(almgren_gamma)
    cfg["num_backtest_trials"] = int(num_trials)

    new_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    if new_hash != st.session_state.last_saved_hash:
        if save_portfolio_config(cfg):
            st.session_state.last_saved_hash = new_hash
            st.caption("🟢 參數已自動即時同步至設定檔")
        else:
            st.warning("⚠️ 設定檔寫入受阻，目前維持記憶體狀態")
    else:
        st.caption("🟢 系統狀態已與設定檔完全同步")


# ==========================================
# PAGE 0: Fancy Special-Effects Portal ("香港股市預測")
# ==========================================
macro_info = cached_macro_regime()
hsi_dist = macro_info.get(
    "hsi_dist_pct",
    (macro_info["hsi_close"] - macro_info["hsi_sma200"]) / (macro_info["hsi_sma200"] + 1e-9),
)

if not st.session_state.entered_main_terminal:
    # Fetch quick live snapshot for the hero showcase cards
    showcase_tickers = [
        ("0700.HK", "騰訊控股"),
        ("0005.HK", "匯豐控股"),
        ("3690.HK", "美團-W"),
        ("0388.HK", "香港交易所"),
    ]
    cards_html = ""
    for tk_code, tk_name in showcase_tickers:
        t_res, _ = cached_stock_trend(tk_code)
        if t_res:
            p_up = t_res["up_prob"] * 100.0
            z_sc = t_res["composite_zscore"]
            p_color = "#34D399" if p_up >= 58 else ("#FB7185" if p_up <= 42 else "#38BDF8")
            outlook_tag = "看多趨勢" if p_up >= 58 else ("看空防禦" if p_up <= 42 else "區間震盪")
            cards_html += f"""
            <div class="portal-stock-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:700; color:#F8FAFC; font-size:0.98rem;">{tk_name}</span>
                    <span class="mono-num" style="color:#94A3B8; font-size:0.80rem;">{tk_code}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px;">
                    <span class="mono-num" style="font-size:1.45rem; color:#FFFFFF;">${t_res['price']:.2f}</span>
                    <span style="font-size:0.78rem; padding:3px 9px; border-radius:999px; background:rgba(56,189,248,0.12); color:{p_color}; border:1px solid {p_color}55; font-weight:600;">
                        {outlook_tag}
                    </span>
                </div>
                <div style="font-size:0.78rem; color:#94A3B8; margin-bottom:4px; display:flex; justify-content:space-between;">
                    <span>AI 上漲機率預測</span>
                    <span class="mono-num" style="color:{p_color};">{p_up:.1f}% ({z_sc:+.2f}σ)</span>
                </div>
                <div class="factor-track">
                    <div style="height:100%; width:{min(100.0, max(10.0, p_up)):.1f}%; background:linear-gradient(90deg, #0EA5E9, {p_color}); border-radius:999px;"></div>
                </div>
            </div>
            """

    hsi_prob_pct = macro_info.get("macro_prob", 0.5) * 100.0
    st.markdown(
        f"""
        <div class="hk-portal-stage">
            <div class="portal-eyebrow">
                <span class="live-dot"></span>
                <span>HKEX AI QUANTITATIVE PREDICTION ENGINE · 機構級多因子量化平台</span>
            </div>
            <div class="portal-main-title">香港股市預測</div>
            <div class="portal-subtitle">
                結合<strong>無未來函數統計校準多因子模型</strong>、<strong>港交所真實微觀結構與市場衝擊滑價</strong>，以及<strong>短中長線全週期歷史勝率檢驗</strong>，將複雜的機構量化分析化繁為簡，為您呈現清晰、直觀、零程式碼干擾的港股智慧決策體驗。
            </div>
            <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:18px; margin-bottom:26px;">
                <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:10px 20px;">
                    <span style="color:#94A3B8; font-size:0.82rem;">恆生指數 HSI：</span>
                    <span class="mono-num" style="font-size:1.1rem; color:#F8FAFC;">{macro_info['hsi_close']:,.0f}</span>
                    <span class="mono-num" style="color:{'#34D399' if hsi_dist>=0 else '#FB7185'}; font-size:0.86rem;"> ({hsi_dist*100:+.1f}% vs 牛熊線)</span>
                </div>
                <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:10px 20px;">
                    <span style="color:#94A3B8; font-size:0.82rem;">大盤多頭情緒溫度：</span>
                    <span class="mono-num" style="font-size:1.1rem; color:#34D399;">{hsi_prob_pct:.1f}%</span>
                </div>
                <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(168,85,247,0.3); border-radius:12px; padding:10px 20px;">
                    <span style="color:#94A3B8; font-size:0.82rem;">當前選中分析標的：</span>
                    <span class="mono-num" style="font-size:1.1rem; color:#38BDF8;">{active_ticker}</span>
                </div>
            </div>
            <div class="portal-orb-row">
                {cards_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2, btn_c3 = st.columns([1.0, 1.4, 1.0])
    with btn_c2:
        if st.button("🚀 立即進入「香港股市預測」四大分析工作台", use_container_width=True, type="primary"):
            st.session_state.entered_main_terminal = True
            st.rerun()

    # Clean, easy-to-understand 4-pillar feature overview
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>⚡ 01 · 日內當沖戰術</span><span style="color:#38BDF8;">即時買賣點</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    嚴格區分「上一根確認收盤價」與「次根開盤實際成交價」，每日 09:30 重置日內均價線，並自動計算港交所印花稅與最佳止損止盈位。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>🔍 02 · 趨勢機率透視</span><span style="color:#34D399;">多因子校準</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    捨棄主觀硬編碼加減分，改以統計機率模型輸出 0%–100% 上漲機率與四大核心因子（動量、價值、低波、流動性）獨立貢獻度。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>📊 03 · 穩健組合調倉</span><span style="color:#F59E0B;">去噪與風控</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    內建停牌遮罩與矩陣去噪技術，自動排除上市未滿 120 日之新股，生成符合港股每手股數與大額拆單提示的實盤調倉單。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g4:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>🎯 04 · 歷史勝率檢驗</span><span style="color:#A855F7;">全週期回測</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    自由切換未來 5 日、10 日、1 個月、3 個月、半年至 1 年預測週期，一鍵檢視歷史方向勝率、五分位數收益階梯與假陽性風險。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


# ==========================================
# Top Macro Regime Live Status Ribbon (Main Terminal)
# ==========================================
regime_badge = (
    '<span class="badge-bull">BULL REGIME 牛市進攻</span>'
    if not macro_info["bear_market"]
    else '<span class="badge-bear">DEFENSIVE 熊市防禦</span>'
)

st.markdown(
    f"""
<div class="macro-ribbon">
    <div class="macro-brand">
        <span class="live-dot"></span>
        <span>香港股市預測 · HKEX ALPHA ENGINE</span>
        <span style="color: #38BDF8; font-weight: 500; font-size: 0.80rem;">INSTITUTIONAL TERMINAL</span>
    </div>
    <div class="macro-pills">
        <div class="macro-item">
            <span>恆生指數 HSI:</span>
            <span class="mono-num">{macro_info['hsi_close']:,.0f}</span>
            <span class="mono-num" style="color: {'#34D399' if hsi_dist >= 0 else '#FB7185'};">({hsi_dist*100:+.1f}% vs 200MA)</span>
        </div>
        <div class="macro-item">
            <span>US 10Y 殖利率:</span>
            <span class="mono-num">{macro_info['tnx_yield']:.2f}%</span>
        </div>
        <div class="macro-item">
            <span>宏觀閘門:</span>
            {regime_badge}
        </div>
        <div class="macro-item">
            <span>當前追蹤:</span>
            <span class="mono-num" style="color: #38BDF8;">{active_ticker} ({interval_choice})</span>
        </div>
        <div class="macro-item">
            <span>去噪與配置:</span>
            <span class="mono-num">{alloc_method.upper()} + {cov_estimator.upper()}</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Main Workspace Tabs (4 Clean, Visual Pages)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    f"⚡ 日內當沖與 VWAP 戰術 ({active_ticker})",
    f"🔍 雙時段趨勢機率與因子透視 ({active_ticker})",
    "📊 穩健組合最佳化與實盤調倉工作台",
    f"🎯 歷史勝率與多週期預測檢驗 ({active_ticker})",
])


# ==========================================
# TAB 1: Intraday Day Trading & Microstructure
# ==========================================
with tab1:
    with st.spinner(f"正在計算 {active_ticker} ({interval_choice}) 無未來函數訊號與港交所微觀滑價..."):
        day_res, day_err = cached_day_trading_signal(
            active_ticker,
            float(min_adv_input),
            interval_choice,
            float(cfg["broker_comm_rate"]),
            float(cfg["min_broker_comm_hkd"]),
            float(cfg["almgren_gamma"]),
        )

    if day_err or not day_res:
        st.error(f"無法載入 {active_ticker}：{day_err}")
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("最新實盤現價", f"${day_res['price']:.2f}", f"決策價 ${day_res['decision_price']:.2f}")
        m2.metric("次根開盤預估撮合價", f"${day_res['execution_price']:.2f}", f"含滑價 {day_res['microstructure']['slippage_bps']:.1f} bps")
        m3.metric("AI 上漲趨勢機率", f"{day_res['up_prob']*100:.1f}%", f"合成 Z {day_res['composite_zscore']:+.2f}σ")
        m4.metric("日內 VWAP 均價 (09:30重置)", f"${day_res['vwap']:.2f}", f"{((day_res['decision_price']-day_res['vwap'])/day_res['vwap'])*100:+.2f}%")
        m5.metric("時段標準化量能分位", f"P{day_res['vol_pct_rank']:.0f}", f"{day_res['vol_ratio']:.2f}x 同刻均量")
        m6.metric(f"每手入場市值 ({day_res['lot_size']}股)", f"${day_res['lot_cost_hkd']:,.0f}", f"法定 Tick ${day_res['tick_size']}")

        b_left, b_right = st.columns([1.1, 1.2])

        with b_left:
            prob_val = day_res["up_prob"]
            sig_badge_cls = "badge-bull" if prob_val >= 0.65 else ("badge-bear" if prob_val <= 0.35 else "badge-warn")
            sfc_badge = (
                '<span class="badge-bull">SFC 可沽空名單</span>'
                if day_res["short_eligible"]
                else '<span class="badge-warn">非可沽空 / 僅限做多</span>'
            )
            mc = day_res["microstructure"]
            st.markdown(
                f"""
            <div class="bento-card">
                <div class="bento-title">
                    <span>🎯 無未來函數決策點位與港交所真實規費試算</span>
                    {sfc_badge}
                </div>
                <div style="margin: 8px 0 14px 0;">
                    <span class="{sig_badge_cls}" style="font-size: 0.90rem; padding: 5px 12px;">
                        {day_res['action']}
                    </span>
                </div>
                <div class="exec-grid">
                    <div class="exec-cell">
                        <div class="exec-label">訊號決策價 (t-1收盤) / 實盤撮合價 (次根開盤)</div>
                        <div class="exec-val">${day_res['decision_price']:.2f} <span style="font-size:0.84rem; color:#38BDF8;">➔ ${day_res['execution_price']:.2f}</span></div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">日內 VWAP 均價 / Wilder's RSI (14)</div>
                        <div class="exec-val" style="color:#F59E0B;">${day_res['vwap']:.2f} <span style="font-size:0.84rem; color:#F8FAFC;">| RSI {day_res['rsi']:.1f}</span></div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">ATR 止損價 (以實際撮合價計算並對齊 Tick)</div>
                        <div class="exec-val" style="color: #FB7185;">${day_res['stop_loss_price']:.2f}</div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">目標止盈價 (1:2 風險報酬比對齊 Tick)</div>
                        <div class="exec-val" style="color: #34D399;">${day_res['take_profit_price']:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:12px; padding-top:10px; border-top:1px solid rgba(148,163,184,0.12); font-size:0.80rem; color:#94A3B8; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <span>印花稅(向上取整): <strong class="mono-num">${mc['stamp_duty_hkd']:.0f}</strong></span>
                    <span>證監+會財+交易所費: <strong class="mono-num">${mc['sfc_levy_hkd']+mc['afrc_levy_hkd']+mc['hkex_fee_hkd']:.2f}</strong></span>
                    <span>經紀佣金: <strong class="mono-num">${mc['broker_commission_hkd']:.1f}</strong></span>
                    <span>衝擊滑價: <strong class="mono-num" style="color:#FBBF24;">{mc['slippage_bps']:.1f} bps</strong></span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with b_right:
            st.markdown(
                render_factor_bars_html(
                    title="🔬 短線五維標準化因子暴露 (Standardized Z-Scores)",
                    subtitle=f"上漲機率: {day_res['up_prob']*100:.1f}% | 合成 Z: {day_res['composite_zscore']:+.2f}σ",
                    factor_dict=day_res["factor_zscores"],
                    max_scale=3.0,
                    unit="σ",
                ),
                unsafe_allow_html=True,
            )

        df_d = day_res["history"]
        fig_3p = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.62, 0.18, 0.20],
            subplot_titles=(
                f"{day_res['ticker']} 價格走勢、每日 09:30 重置之日內 VWAP 與 ±1.5 ATR 波動通道 ({day_res['interval']})",
                "時段成交量與 20 期均量",
                "Wilder's RSI (14) 動能強弱",
            ),
        )

        fig_3p.add_trace(
            go.Candlestick(
                x=df_d.index,
                open=df_d["Open"],
                high=df_d["High"],
                low=df_d["Low"],
                close=df_d["Close"],
                name="K線",
                increasing_line_color="#10B981",
                decreasing_line_color="#F43F5E",
            ),
            row=1,
            col=1,
        )

        if "VWAP" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["VWAP"],
                    mode="lines",
                    name="VWAP 均價線 (09:30重置)",
                    line=dict(color="#F59E0B", width=2.2),
                ),
                row=1,
                col=1,
            )

        if "ATR_Upper" in df_d.columns and "ATR_Lower" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["ATR_Upper"],
                    mode="lines",
                    name="+1.5 ATR 通道",
                    line=dict(color="rgba(56, 189, 248, 0.35)", width=1, dash="dot"),
                ),
                row=1,
                col=1,
            )
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["ATR_Lower"],
                    mode="lines",
                    name="-1.5 ATR 通道",
                    line=dict(color="rgba(56, 189, 248, 0.35)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(56, 189, 248, 0.05)",
                ),
                row=1,
                col=1,
            )

        vol_colors = np.where(df_d["Close"] >= df_d["Open"], "#10B981", "#F43F5E")
        fig_3p.add_trace(
            go.Bar(x=df_d.index, y=df_d["Volume"], name="成交量", marker_color=vol_colors, opacity=0.85),
            row=2,
            col=1,
        )
        if "Vol_MA20" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["Vol_MA20"],
                    mode="lines",
                    name="20期均量",
                    line=dict(color="#38BDF8", width=1.4),
                ),
                row=2,
                col=1,
            )

        if "RSI" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["RSI"],
                    mode="lines",
                    name="RSI (14)",
                    line=dict(color="#A855F7", width=2),
                ),
                row=3,
                col=1,
            )
            fig_3p.add_hline(y=70, line_dash="dash", line_color="rgba(244,63,94,0.5)", row=3, col=1)
            fig_3p.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.5)", row=3, col=1)

        fig_3p.update_layout(
            height=620,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,25,0.72)",
            font=dict(color="#CBD5E1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=15, t=35, b=15),
        )
        fig_3p.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        if day_res["interval"] in ("5m", "15m"):
            fig_3p.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                    dict(bounds=[12, 13], pattern="hour"),
                ]
            )
        fig_3p.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        st.plotly_chart(fig_3p, use_container_width=True)


# ==========================================
# TAB 2: Calibrated Multi-Factor Probability & VIF Matrix
# ==========================================
with tab2:
    with st.spinner(f"正在計算 {active_ticker} 的 IC 加權多因子機率模型與共線性診斷..."):
        trend_res, trend_err = cached_stock_trend(active_ticker)

    if trend_err or not trend_res:
        st.error(f"無法分析 {active_ticker}：{trend_err}")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("最新現價 / 次日撮合價", f"${trend_res['price']:.2f}", f"撮合 ${trend_res['execution_price']:.2f}")
        c2.metric("趨勢上漲機率 P(Up)", f"{trend_res['up_prob']*100:.1f}%", f"合成 Z {trend_res['composite_zscore']:+.2f}σ")
        c3.metric("短期戰術上漲機率", f"{trend_res['short_term_prob']*100:.1f}%", f"20D Z {trend_res['z_20d']:+.2f}σ")
        c4.metric("長期戰略上漲機率", f"{trend_res['long_term_prob']*100:.1f}%", f"200MA {trend_res['dist_200ma']*100:+.1f}%")
        c5.metric("滾動股息殖利率", f"{trend_res['dividend_yield']*100:.2f}%", "前復權已計入")
        c6.metric("股東權益報酬 (ROE)", f"{trend_res['roe']*100:.1f}%", "財報質量因子")

        t2_left, t2_right = st.columns([1.15, 1.15])
        with t2_left:
            z_dict_display = {
                row["factor_name"]: float(row["z_score"])
                for row in trend_res["factor_exposure_matrix"]
            }
            st.markdown(
                render_factor_bars_html(
                    title="🏛️ 資訊係數 (IC) 加權四大核心因子標準化暴露",
                    subtitle=f"{trend_res['trend']}",
                    factor_dict=z_dict_display,
                    max_scale=3.0,
                    unit="σ",
                ),
                unsafe_allow_html=True,
            )
        with t2_right:
            bullets_html = "".join(
                f'<div style="padding: 6px 0; border-bottom: 1px solid rgba(148,163,184,0.08); font-size: 0.84rem; color: #CBD5E1;">{r}</div>'
                for r in trend_res["reasons"]
            )
            st.markdown(
                f"""
            <div class="bento-card">
                <div class="bento-title">
                    <span>📋 邏輯回歸趨勢機率與無未來函數診斷明細</span>
                    <span class="mono-num" style="color:#34D399;">P(Up): {trend_res['up_prob']*100:.1f}%</span>
                </div>
                {bullets_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Factor Exposure & Collinearity Diagnostics Matrix
        exp_col, corr_col = st.columns([1.35, 0.95])
        with exp_col:
            st.markdown("#### 📐 標準化因子暴露矩陣與共線性診斷 (VIF 變異數膨脹因子)")
            df_exp = pd.DataFrame(trend_res["factor_exposure_matrix"])
            st.dataframe(
                df_exp[[
                    "factor_name",
                    "z_score",
                    "rank_ic",
                    "ic_weight_pct",
                    "marginal_contribution_z",
                    "vif",
                    "collinearity_status",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "factor_name": st.column_config.TextColumn("核心因子名稱"),
                    "z_score": st.column_config.NumberColumn("標準化 Z-Score", format="%+.2fσ"),
                    "rank_ic": st.column_config.NumberColumn("滾動 Rank IC", format="%+.3f"),
                    "ic_weight_pct": st.column_config.ProgressColumn("IC 校準權重", format="%.1f%%", min_value=0, max_value=100),
                    "marginal_contribution_z": st.column_config.NumberColumn("邊際貢獻度", format="%+.3fσ"),
                    "vif": st.column_config.NumberColumn("共線性 VIF", format="%.2f"),
                    "collinearity_status": st.column_config.TextColumn("正交診斷結果"),
                },
            )
        with corr_col:
            st.markdown("#### 🔗 因子間相關性診斷矩陣")
            f_corr = trend_res["factor_corr_matrix"].copy()
            short_names = ["動量", "價值質量", "低波品質", "流動資金"]
            f_corr.index = short_names
            f_corr.columns = short_names
            fig_fcorr = px.imshow(
                f_corr, text_auto=".2f", color_continuous_scale="Tealrose", zmin=-1, zmax=1
            )
            fig_fcorr.update_layout(
                height=245,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fcorr, use_container_width=True)

        df_h = trend_res["history"]
        fig_t2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{trend_res['ticker']} 前復權股價走勢 + 50MA / 200MA 牛熊分界線 (消除除息跳空失真)",
                "滾動 20 日波動率標準化動量 Z-Score (±3σ 統計校準區間)",
            ),
        )
        fig_t2.add_trace(
            go.Candlestick(
                x=df_h.index,
                open=df_h["Open"],
                high=df_h["High"],
                low=df_h["Low"],
                close=df_h["Close"],
                name="前復權 K線",
                increasing_line_color="#10B981",
                decreasing_line_color="#F43F5E",
            ),
            row=1,
            col=1,
        )

        if "SMA50" in df_h.columns:
            fig_t2.add_trace(
                go.Scatter(
                    x=df_h.index,
                    y=df_h["SMA50"],
                    mode="lines",
                    name="SMA 50 (中線)",
                    line=dict(color="#38BDF8", width=1.8),
                ),
                row=1,
                col=1,
            )
        if "SMA200" in df_h.columns:
            fig_t2.add_trace(
                go.Scatter(
                    x=df_h.index,
                    y=df_h["SMA200"],
                    mode="lines",
                    name="SMA 200 (牛熊線)",
                    line=dict(color="#F59E0B", width=2.2),
                ),
                row=1,
                col=1,
            )

        if "Z_20D" in df_h.columns:
            z_colors = np.where(df_h["Z_20D"] >= 0, "#10B981", "#F43F5E")
            fig_t2.add_trace(
                go.Bar(
                    x=df_h.index,
                    y=df_h["Z_20D"],
                    name="20日動量 Z-Score",
                    marker_color=z_colors,
                    opacity=0.85,
                ),
                row=2,
                col=1,
            )
            fig_t2.add_hline(y=1.5, line_dash="dot", line_color="rgba(16,185,129,0.5)", row=2, col=1)
            fig_t2.add_hline(y=-1.5, line_dash="dot", line_color="rgba(244,63,94,0.5)", row=2, col=1)

        fig_t2.update_layout(
            height=540,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,25,0.72)",
            font=dict(color="#CBD5E1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=15, t=35, b=15),
        )
        fig_t2.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        fig_t2.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        st.plotly_chart(fig_t2, use_container_width=True)


# ==========================================
# TAB 3: Portfolio Workbench, Denoised Cov & Blotter
# ==========================================
with tab3:
    st.markdown("""
    <div class="bento-card" style="padding: 14px 20px; margin-bottom: 14px;">
        <div class="bento-title" style="margin-bottom: 4px;">
            <span>💼 投資組合持倉管理與多標的配置工作台</span>
            <span style="color: #38BDF8;">支援點選預設範本、一鍵加入任意港股，或雙擊下方表格直接修改</span>
        </div>
        <div style="font-size:0.84rem; color:#94A3B8; margin-top:4px;">
            💡 <strong>如何更換持倉組合？</strong> 您可以：① 點選下方 4 組<strong>精選範本</strong>一鍵切換；② 在輸入框輸入代號（如 1810.HK）並點擊<strong>加入持倉</strong>；③ 展開下方表格<strong>直接雙擊代號或股數修改</strong>，或勾選刪除。修改將自動即時同步。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🚀 快速套用精選投資組合範本 (一鍵切換)")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🌐 港股科技巨頭\n(0700, 9988, 3690, 1810)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0700.HK", "name": "騰訊控股", "lot_size": 100, "shares": 300, "entry_price": 380.0, "short_eligible": True},
                {"ticker": "9988.HK", "name": "阿里巴巴-W", "lot_size": 100, "shares": 500, "entry_price": 85.0, "short_eligible": True},
                {"ticker": "3690.HK", "name": "美團-W", "lot_size": 100, "shares": 400, "entry_price": 130.0, "short_eligible": True},
                {"ticker": "1810.HK", "name": "小米集團-W", "lot_size": 200, "shares": 2000, "entry_price": 25.0, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col2:
        if st.button("🏛️ 藍籌金融核心\n(0005, 0388, 1299, 0939)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0005.HK", "name": "匯豐控股", "lot_size": 400, "shares": 1200, "entry_price": 68.0, "short_eligible": True},
                {"ticker": "0388.HK", "name": "香港交易所", "lot_size": 100, "shares": 300, "entry_price": 280.0, "short_eligible": True},
                {"ticker": "1299.HK", "name": "友邦保險", "lot_size": 200, "shares": 1000, "entry_price": 62.0, "short_eligible": True},
                {"ticker": "0939.HK", "name": "建設銀行", "lot_size": 1000, "shares": 10000, "entry_price": 5.8, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col3:
        if st.button("🚗 新能源與製造\n(1211, 2015, 0175, 2382)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "1211.HK", "name": "比亞迪股份", "lot_size": 500, "shares": 500, "entry_price": 260.0, "short_eligible": True},
                {"ticker": "2015.HK", "name": "理想汽車-W", "lot_size": 100, "shares": 600, "entry_price": 100.0, "short_eligible": True},
                {"ticker": "0175.HK", "name": "吉利汽車", "lot_size": 1000, "shares": 4000, "entry_price": 12.0, "short_eligible": True},
                {"ticker": "2382.HK", "name": "舜宇光學", "lot_size": 100, "shares": 500, "entry_price": 55.0, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col4:
        if st.button("🛡️ 高息防禦公用\n(0941, 0883, 0002, 0003)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0941.HK", "name": "中國移動", "lot_size": 500, "shares": 1500, "entry_price": 72.0, "short_eligible": True},
                {"ticker": "0883.HK", "name": "中國海洋石油", "lot_size": 1000, "shares": 3000, "entry_price": 18.0, "short_eligible": True},
                {"ticker": "0002.HK", "name": "中電控股", "lot_size": 500, "shares": 1500, "entry_price": 65.0, "short_eligible": True},
                {"ticker": "0003.HK", "name": "中華煤氣", "lot_size": 1000, "shares": 5000, "entry_price": 6.2, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()

    qa_c1, qa_c2, qa_c3 = st.columns([1.5, 1.0, 1.0])
    with qa_c1:
        add_ticker_input = st.text_input("➕ 快速新增標的代號 (例如 1810.HK 或 0941)", placeholder="輸入港股代號如 1810.HK", key="quick_add_input")
    with qa_c2:
        add_shares_input = st.number_input("持有股數", min_value=0, value=500, step=100, key="quick_add_shares")
    with qa_c3:
        st.write("")
        st.write("")
        if st.button("加入持倉組合", type="secondary", use_container_width=True):
            if add_ticker_input.strip():
                norm_add = normalize_hk_ticker(add_ticker_input.strip())
                curr_tickers = [h.get("ticker", "") for h in cfg.get("holdings", [])]
                if norm_add not in curr_tickers:
                    cfg["holdings"].append({
                        "ticker": norm_add,
                        "name": norm_add,
                        "lot_size": HK_DEFAULT_LOT_SIZES.get(norm_add, 100),
                        "shares": int(add_shares_input),
                        "entry_price": 0.0,
                        "short_eligible": is_hk_short_eligible(norm_add),
                    })
                    save_portfolio_config(cfg)
                    st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
                    st.rerun()
                else:
                    st.toast(f"{norm_add} 已存在於持倉中！")

    with st.expander("✏️ 詳細持倉數據表 (支援雙擊儲存格修改代號/股數，或勾選刪除)", expanded=True):
        df_edit_in = pd.DataFrame(cfg.get("holdings", []))
        if df_edit_in.empty:
            df_edit_in = pd.DataFrame(columns=["ticker", "name", "lot_size", "shares"])
        else:
            for col in ["ticker", "name", "lot_size", "shares"]:
                if col not in df_edit_in.columns:
                    df_edit_in[col] = ""

        edited_df = st.data_editor(
            df_edit_in[["ticker", "name", "lot_size", "shares"]],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("港股代號 (如 0700.HK)", required=True),
                "name": st.column_config.TextColumn("股票名稱", required=True),
                "lot_size": st.column_config.NumberColumn("每手股數 (Lot Size)", min_value=1, step=50, required=True),
                "shares": st.column_config.NumberColumn("目前持有股數", min_value=0, step=100, required=True),
            },
            key="holdings_editor",
        )

        existing_map = {normalize_hk_ticker(h.get("ticker", "")): h for h in cfg.get("holdings", [])}
        new_holdings = []
        for _, row in edited_df.iterrows():
            t_str = str(row.get("ticker", "")).strip()
            if not t_str:
                continue
            norm_tk = normalize_hk_ticker(t_str)
            base_item = dict(existing_map.get(norm_tk, {}))
            base_item.update({
                "ticker": norm_tk,
                "name": str(row.get("name", norm_tk)).strip(),
                "lot_size": int(row.get("lot_size") or HK_DEFAULT_LOT_SIZES.get(norm_tk, 100)),
                "shares": int(row.get("shares") or 0),
            })
            new_holdings.append(base_item)

        if new_holdings != cfg.get("holdings", []):
            cfg["holdings"] = new_holdings
            if save_portfolio_config(cfg):
                st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)

    holdings_serialized = json.dumps(cfg["holdings"], ensure_ascii=False, sort_keys=True)

    with st.spinner("正在執行停牌遮罩過濾、Ledoit-Wolf / RMT 矩陣去噪與無未來函數滾動回測..."):
        health_res = cached_portfolio_health(
            holdings_serialized,
            float(total_capital_hkd),
            float(rf_rate_pct) / 100.0,
            float(max_pos_pct) / 100.0,
            float(min_adv_input),
            alloc_method,
            int(num_trials),
            cov_estimator,
            float(max_turnover_pct) / 100.0,
            float(cfg["broker_comm_rate"]),
            float(cfg["min_broker_comm_hkd"]),
            float(cfg["almgren_gamma"]),
        )

    if not health_res:
        st.error("無法計算組合風險，請確認持倉表格中的港股代號是否有效且具備至少 120 個有效交易日。")
    else:
        bt = health_res["backtest"]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("現有持倉總市值", f"${health_res['total_value']:,.0f}", f"目標池 ${total_capital_hkd:,.0f}")
        k2.metric("組合風控健康度", f"{health_res['health_score']}/100", health_res["health_status"].split(" ")[1])
        if bt:
            k3.metric("滾動回測幾何年化", f"{bt['cagr']*100:+.2f}%", f"總回報 {bt['total_return']*100:+.1f}%")
            k4.metric(f"夏普比率 (Rf={rf_rate_pct:.1f}%)", f"{bt['sharpe']:.2f}", f"波動率 {bt['ann_volatility']*100:.1f}%")
            k5.metric("DSR 抗過擬合信心度", f"{bt['dsr_prob']*100:.1f}%", f"p={bt['dsr_pvalue']:.3f}")
            k6.metric("最大回撤 / 1D CVaR", f"{bt['max_drawdown']*100:.1f}%", f"CVaR {health_res['cvar_95']*100:.2f}%")
        else:
            k3.metric("組合年化波動率", f"{health_res['ann_volatility']*100:.2f}%")
            k4.metric("現有夏普比率", f"{health_res['sharpe_ratio']:.2f}")
            k5.metric("單日尾部 CVaR (95%)", f"{health_res['cvar_95']*100:.2f}%")
            k6.metric("去噪後平均相關性", f"{health_res['avg_correlation']:.2f}")

        chart_left, chart_right = st.columns([1.35, 1.0])

        with chart_left:
            if bt:
                fig_bt = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.68, 0.32],
                    subplot_titles=(
                        f"樣本外滾動回測淨值 ({method_options[alloc_method]} | 嚴格 t-1 窗口 + 港交所規費與衝擊滑價)",
                        "水下回撤曲線 (Underwater Drawdown %)",
                    ),
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["strategy_nav"],
                        mode="lines",
                        name=f"{alloc_method.upper()} 策略淨值",
                        line=dict(color="#10B981", width=2.4),
                    ),
                    row=1,
                    col=1,
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["benchmark_nav"],
                        mode="lines",
                        name="等權重基準 (Benchmark)",
                        line=dict(color="#94A3B8", width=1.5, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["drawdown_series"] * 100.0,
                        mode="lines",
                        name="策略水下回撤 (%)",
                        line=dict(color="#F43F5E", width=1.6),
                        fill="tozeroy",
                        fillcolor="rgba(244, 63, 94, 0.18)",
                    ),
                    row=2,
                    col=1,
                )

                fig_bt.update_layout(
                    height=460,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,25,0.72)",
                    font=dict(color="#CBD5E1", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=15, r=15, t=35, b=15),
                )
                fig_bt.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
                fig_bt.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
                st.plotly_chart(fig_bt, use_container_width=True)

        with chart_right:
            sub_t1, sub_t2 = st.tabs(["⚖️ 當前 vs 目標權重 (含換手約束)", "🔥 去噪後資產相關性矩陣"])
            with sub_t1:
                curr_w = health_res["weights"]
                opt_w = health_res["optimal_weights"]
                tickers_all = list(opt_w.keys())
                df_w = pd.DataFrame({
                    "Ticker": tickers_all,
                    "當前權重 (%)": [curr_w.get(t, 0.0) * 100.0 for t in tickers_all],
                    "目標權重 (%)": [opt_w.get(t, 0.0) * 100.0 for t in tickers_all],
                })
                fig_w = go.Figure()
                fig_w.add_trace(
                    go.Bar(
                        y=df_w["Ticker"],
                        x=df_w["當前權重 (%)"],
                        orientation="h",
                        name="當前持倉權重",
                        marker_color="#64748B",
                    )
                )
                fig_w.add_trace(
                    go.Bar(
                        y=df_w["Ticker"],
                        x=df_w["目標權重 (%)"],
                        orientation="h",
                        name=f"{alloc_method.upper()} 目標權重",
                        marker_color="#38BDF8",
                    )
                )
                fig_w.update_layout(
                    barmode="group",
                    height=395,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,25,0.72)",
                    font=dict(color="#CBD5E1", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(fig_w, use_container_width=True)

            with sub_t2:
                corr_df = health_res["corr_matrix"]
                fig_heat = px.imshow(
                    corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
                )
                fig_heat.update_layout(
                    height=395,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1", size=11),
                    margin=dict(l=10, r=10, t=20, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        blotter_col, diag_col = st.columns([1.55, 0.85])

        with blotter_col:
            st.markdown("#### 🛒 港交所實盤調倉執行單 (含法定規費、Almgren-Chriss 滑價與 >5% ADV 拆單提示)")
            exec_df = pd.DataFrame(health_res["execution_orders"])
            exec_df["target_weight_pct"] = exec_df["target_weight"] * 100.0
            st.dataframe(
                exec_df[[
                    "ticker",
                    "name",
                    "price",
                    "lot_size",
                    "target_weight_pct",
                    "current_shares",
                    "target_shares",
                    "action",
                    "estimated_turnover",
                    "statutory_fees_hkd",
                    "slippage_bps",
                    "total_friction_hkd",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("代號"),
                    "name": st.column_config.TextColumn("名稱"),
                    "price": st.column_config.NumberColumn("Tick對齊現價", format="$%.2f"),
                    "lot_size": st.column_config.NumberColumn("每手股數", format="%d"),
                    "target_weight_pct": st.column_config.ProgressColumn(
                        "目標權重", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "current_shares": st.column_config.NumberColumn("當前股數", format="%d"),
                    "target_shares": st.column_config.NumberColumn("目標股數", format="%d"),
                    "action": st.column_config.TextColumn("實盤調倉指令與拆單建議"),
                    "estimated_turnover": st.column_config.NumberColumn("預估交易額", format="$%,.0f"),
                    "statutory_fees_hkd": st.column_config.NumberColumn("法定規費+佣金", format="$%,.1f"),
                    "slippage_bps": st.column_config.NumberColumn("預估滑價 (bps)", format="%.1f"),
                    "total_friction_hkd": st.column_config.NumberColumn("總摩擦成本", format="$%,.1f"),
                },
            )

        with diag_col:
            warn_items_html = "".join(
                f'<div style="padding: 8px 0; border-bottom: 1px solid rgba(148,163,184,0.10); font-size: 0.84rem; color: #E2E8F0;">{w}</div>'
                for w in health_res["warnings"]
            )
            st.markdown(
                f"""
            <div class="bento-card" style="margin-top: 28px;">
                <div class="bento-title">
                    <span>🛡️ 機構級風控、停牌遮罩與 DSR 審計</span>
                    <span class="mono-num" style="color: #38BDF8;">{health_res['health_score']}/100</span>
                </div>
                {warn_items_html}
            </div>
            """,
                unsafe_allow_html=True,
            )


# ==========================================
# TAB 4: Historical Prediction Accuracy & Multi-Horizon Validation (Task 3)
# ==========================================
with tab4:
    st.markdown(
        """
        <div class="bento-card" style="padding: 14px 20px; margin-bottom: 16px;">
            <div class="bento-title" style="margin-bottom:4px;">
                <span>🎛️ 歷史預測準確度與多週期回測配置面板 (Short-Term to Long-Term Horizons)</span>
                <span style="color:#34D399;">嚴格 t-1 訊號決策 ➔ 次日開盤撮合 (零未來函數)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns([1.3, 1.0, 1.0, 1.0])
    horizon_presets = {
        5: "短線戰術：未來 5 個交易日 (5D)",
        10: "短線戰術：未來 10 個交易日 (10D)",
        21: "中線波段：未來 1 個月 (21D)",
        63: "中線波段：未來 3 個月 (63D)",
        126: "長線戰略：未來 6 個月 (126D)",
        252: "長線戰略：未來 1 年 (252D)",
    }
    with cfg_c1:
        preset_k = st.selectbox(
            "選擇標準前瞻評估週期",
            options=list(horizon_presets.keys()),
            format_func=lambda k: horizon_presets[k],
            index=2,
        )
        custom_k = st.slider(
            "或自由滑動微調前瞻天數 k (交易日)",
            min_value=3,
            max_value=252,
            value=int(preset_k),
            step=1,
        )
    with cfg_c2:
        hist_len_map = {
            "6mo": "近 6 個月",
            "1y": "近 1 年",
            "3y": "近 3 年",
            "5y": "近 5 年",
        }
        hist_period_sel = st.selectbox(
            "歷史回測數據長度",
            options=list(hist_len_map.keys()),
            format_func=lambda k: hist_len_map[k],
            index=2,
        )
    with cfg_c3:
        atr_stop_mult_sel = st.slider(
            "ATR 動態停損乘數 (倍)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.25,
        )
    with cfg_c4:
        sig_thresh_sel = st.slider(
            "訊號觸發門檻閥值 (|Z| ≥ σ)",
            min_value=0.0,
            max_value=1.5,
            value=0.25,
            step=0.05,
        )

    with st.spinner(f"正在對 {active_ticker} 進行 {hist_len_map[hist_period_sel]}、前瞻 {custom_k} 個交易日之向量化歷史準確度檢驗..."):
        acc_res, acc_err = cached_historical_accuracy(
            ticker=active_ticker,
            horizon=int(custom_k),
            history_period=hist_period_sel,
            atr_stop_mult=float(atr_stop_mult_sel),
            signal_z_threshold=float(sig_thresh_sel),
            comm_rate=float(cfg["broker_comm_rate"]),
            min_comm=float(cfg["min_broker_comm_hkd"]),
            gamma=float(cfg["almgren_gamma"]),
        )

    if acc_err or not acc_res:
        st.error(f"無法完成歷史準確度驗證：{acc_err}")
    else:
        # Row 1: 6 Core Institutional Accuracy KPI Cards
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric(
            "歷史方向勝率 (Hit Rate)",
            f"{acc_res['hit_rate']*100:.1f}%",
            f"有效訊號 {acc_res['sample_size']} 次",
        )
        a2.metric(
            "平均 Rank IC (秩相關)",
            f"{acc_res['mean_ic']:+.3f}",
            f"全樣本 IC {acc_res['overall_rank_ic']:+.3f}",
        )
        a3.metric(
            "IC 資訊比率 (IC-IR)",
            f"{acc_res['ic_ir']:+.2f}",
            "顯著預測力" if abs(acc_res["ic_ir"]) >= 0.5 else "溫和預測力",
        )
        a4.metric(
            "年化超額回報 (vs 恆指)",
            f"{acc_res['annualized_alpha']*100:+.2f}%",
            f"策略年化 {acc_res['strategy_ann_return']*100:+.1f}%",
        )
        a5.metric(
            "獲利因子 (Profit Factor)",
            f"{acc_res['profit_factor']:.2f}",
            f"盈虧比 {acc_res['win_loss_ratio']:.2f}x",
        )
        a6.metric(
            "平均實際持倉週期",
            f"{acc_res['avg_holding_days']:.1f} 日",
            f"單筆期望 {acc_res['expected_return_pct']:+.2f}%",
        )

        # Row 2: Quantile Monotonicity Bar Chart + Cumulative Quantile Curves
        q_left, q_right = st.columns([1.05, 1.25])
        q_df = pd.DataFrame(acc_res["quantile_summary"])

        with q_left:
            bar_colors = ["#F43F5E", "#FB923C", "#94A3B8", "#38BDF8", "#10B981"]
            fig_qbar = go.Figure(
                go.Bar(
                    x=q_df["quantile"],
                    y=q_df["mean_forward_return_pct"],
                    marker_color=bar_colors,
                    text=[f"{v:+.2f}%" for v in q_df["mean_forward_return_pct"]],
                    textposition="outside",
                )
            )
            fig_qbar.update_layout(
                title=f"📊 五分位數單調性檢驗 (Q1 至 Q5 前瞻 {acc_res['forward_horizon']} 日平均收益 | 多空差 {acc_res['q5_q1_spread_pct']:+.2f}%)",
                height=390,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,25,0.72)",
                font=dict(color="#CBD5E1", size=11),
                yaxis_title=f"前瞻 {acc_res['forward_horizon']} 日平均收益率 (%)",
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_qbar.update_xaxes(showgrid=False)
            fig_qbar.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            st.plotly_chart(fig_qbar, use_container_width=True)

        with q_right:
            fig_qnav = go.Figure()
            q_palette = {
                "Q1 (極空)": "#F43F5E",
                "Q2 (偏空)": "#FB923C",
                "Q3 (中性)": "#94A3B8",
                "Q4 (偏多)": "#38BDF8",
                "Q5 (極多)": "#10B981",
            }
            for q_name, nav_s in acc_res["quantile_cum_curves"].items():
                fig_qnav.add_trace(
                    go.Scatter(
                        x=nav_s.index,
                        y=nav_s.values,
                        mode="lines",
                        name=q_name,
                        line=dict(color=q_palette.get(q_name, "#38BDF8"), width=2.4 if q_name in ("Q5 (極多)", "Q1 (極空)") else 1.4),
                    )
                )
            fig_qnav.update_layout(
                title=f"📈 分位數分組累積淨值曲線 (單調性秩相關係數 = {acc_res['monotonicity_score']:+.2f})",
                height=390,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,25,0.72)",
                font=dict(color="#CBD5E1", size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="累積淨值 (基準 = 1.0)",
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_qnav.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            fig_qnav.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            st.plotly_chart(fig_qnav, use_container_width=True)

        # Row 3: Scatter Prediction Plot (with Regression Line & R^2) + Confusion Matrix & MAE/MFE
        sc_left, sc_right = st.columns([1.15, 1.15])
        sc_df = acc_res["scatter_df"]
        reg = acc_res["regression"]

        with sc_left:
            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=sc_df["pred_score"],
                    y=sc_df["forward_return_pct"],
                    mode="markers",
                    name="歷史樣本點",
                    marker=dict(
                        size=6.5,
                        color=sc_df["pred_prob_pct"],
                        colorscale="Tealgrn",
                        opacity=0.72,
                        showscale=True,
                        colorbar=dict(title="上漲機率%"),
                    ),
                    text=sc_df["date"],
                    hovertemplate="日期: %{text}<br>預測得分: %{x:.1f}<br>實際前瞻漲跌: %{y:+.2f}%<extra></extra>",
                )
            )
            x_line = np.linspace(float(sc_df["pred_score"].min()), float(sc_df["pred_score"].max()), 50)
            y_line = reg["slope"] * x_line + reg["intercept"]
            fig_sc.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"線性回歸趨勢線 (R² = {reg['r_squared']:.3f})",
                    line=dict(color="#F59E0B", width=2.5, dash="dash"),
                )
            )
            fig_sc.update_layout(
                title=f"🎯 模型預測得分 vs 未來 {acc_res['forward_horizon']} 日實際漲跌幅散點圖 (R² = {reg['r_squared']:.3f})",
                height=395,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,25,0.72)",
                font=dict(color="#CBD5E1", size=11),
                xaxis_title="模型預測綜合得分 (-100 至 +100)",
                yaxis_title=f"未來 {acc_res['forward_horizon']} 日實際漲跌幅 (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_sc.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            fig_sc.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            st.plotly_chart(fig_sc, use_container_width=True)

        with sc_right:
            cm_tab1, cm_tab2 = st.tabs(["🛡️ 持倉最大有利/不利偏離 (MAE / MFE)", "🧩 訊號混淆矩陣與假陽性率 (FPR)"])
            with cm_tab1:
                fig_exc = go.Figure()
                fig_exc.add_trace(
                    go.Histogram(
                        x=sc_df["mfe_pct"],
                        name="MFE 最大有利偏離 (潛在浮盈 %)",
                        marker_color="#10B981",
                        opacity=0.70,
                    )
                )
                fig_exc.add_trace(
                    go.Histogram(
                        x=sc_df["mae_pct"],
                        name="MAE 最大不利回撤 (潛在浮虧 %)",
                        marker_color="#F43F5E",
                        opacity=0.70,
                    )
                )
                fig_exc.update_layout(
                    barmode="overlay",
                    height=330,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,25,0.72)",
                    font=dict(color="#CBD5E1", size=11),
                    xaxis_title="持倉期間最大價格偏離幅度 (%)",
                    yaxis_title="樣本頻次",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(fig_exc, use_container_width=True)

            with cm_tab2:
                cm = acc_res["confusion_matrix"]
                st.markdown(
                    f"""
                    <div class="bento-card" style="margin-top:8px;">
                        <div class="bento-title">
                            <span>🔍 多空預測混淆矩陣與停損觸發統計</span>
                            <span class="mono-num" style="color:#FBBF24;">假陽性率 FPR: {cm['fpr']*100:.1f}%</span>
                        </div>
                        <div class="exec-grid">
                            <div class="exec-cell">
                                <div class="exec-label">真陽性 (TP · 看多且獲利未停損)</div>
                                <div class="exec-val" style="color:#34D399;">{cm['tp']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(精確率 {cm['precision']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">假陽性 (FP · 看多但虧損或觸發停損)</div>
                                <div class="exec-val" style="color:#FB7185;">{cm['fp']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(FPR {cm['fpr']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">真陰性 (TN · 看空成功規避下跌)</div>
                                <div class="exec-val" style="color:#38BDF8;">{cm['tn']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(停損率 {cm['stop_trigger_rate']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">假陰性 (FN · 看空但標的逆勢上漲)</div>
                                <div class="exec-val" style="color:#F59E0B;">{cm['fn']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(召回率 {cm['recall']*100:.1f}%)</span></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )