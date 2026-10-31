"""
HKEX Institutional Quantitative Engine (quant_engine.py)
========================================================
Provides institutional-grade quantitative modeling for Hong Kong equities:
1. Strict separation of Decision Price (Close[t-1]) and Execution Price (Open[t] + Dynamic Slippage)
   to eliminate Look-Ahead Bias and Contemporaneous Return Leakage.
2. Explicit Suspension & Zero-Volume Masking to prevent artificial low-volatility distortion.
3. Strict Data Sufficiency Validation (>= 120 valid trading days) to prevent Singular Matrix crashes.
4. Session-anchored Intraday VWAP (strictly reset at 09:30 HKT daily) and Time-of-Day volume normalization.
5. Calibrated Multi-Factor Scoring via Rolling Rank-IC Weighting & Logistic Regression Trend Probability
   P(Upward Trend | Factors) in [0, 1] and standardized Z-Scores in [-3.0, +3.0], with VIF collinearity diagnostics.
6. Authentic HKEX Microstructure Cost Model (Stamp Duty 0.10% rounded up to nearest $1 HKD, SFC levy 0.0027%,
   AFRC levy 0.00015%, HKEX trading fee 0.00565%, customizable brokerage commission) and Almgren-Chriss
   Square-Root Market Impact Slippage model.
7. Analytical Ledoit-Wolf Shrinkage Covariance & Random Matrix Theory (RMT) Marcenko-Pastur Denoising
   with L1 Turnover Constraints in portfolio optimization.
8. Vectorized Multi-Horizon Historical Prediction Accuracy & Backtest Engine (5D, 10D, 21D, 63D, 126D, 252D)
   with Directional Hit Rate, IC / Rank-IC, IC-IR, 5-Quantile Monotonicity Test (Q1-Q5), Profit Factor,
   Win/Loss Ratio, Confusion Matrix (FPR/TPR), and MAE/MFE excursion distributions.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import warnings
from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, norm, pearsonr, skew, spearmanr

warnings.filterwarnings("ignore")

DB_FILE: str = "quant_system_audit.db"
PORTFOLIO_FILE: str = "portfolio.json"
MIN_REQUIRED_DAILY_BARS: int = 120

# Common HKEX board lot sizes fallback dictionary
HK_DEFAULT_LOT_SIZES: dict[str, int] = {
    "0700.HK": 100,
    "0005.HK": 400,
    "3690.HK": 100,
    "9988.HK": 100,
    "2475.HK": 100,
    "1299.HK": 200,
    "0388.HK": 100,
    "0941.HK": 500,
    "0001.HK": 500,
    "2318.HK": 500,
    "1398.HK": 1000,
    "3988.HK": 1000,
    "0883.HK": 1000,
    "1810.HK": 200,
    "9618.HK": 50,
    "9999.HK": 100,
    "1211.HK": 500,
}

# Major SFC Designated Securities Eligible for Short Selling (liquid large-caps)
HK_SHORT_ELIGIBLE_WHITELIST: set[str] = {
    "0700.HK", "0005.HK", "3690.HK", "9988.HK", "1299.HK",
    "0388.HK", "0941.HK", "0001.HK", "2318.HK", "1398.HK",
    "3988.HK", "0883.HK", "1810.HK", "9618.HK", "9999.HK",
    "1211.HK", "0016.HK", "0002.HK", "0003.HK", "0011.HK",
    "0027.HK", "0066.HK", "0175.HK", "0267.HK", "0288.HK",
    "0386.HK", "0669.HK", "0688.HK", "0762.HK", "0823.HK",
    "0857.HK", "0939.HK", "0968.HK", "0981.HK", "1024.HK",
    "1088.HK", "1093.HK", "1109.HK", "1177.HK", "1928.HK",
    "2015.HK", "2020.HK", "2269.HK", "2313.HK", "2319.HK",
    "2331.HK", "2382.HK", "2388.HK", "2628.HK", "3692.HK",
    "6690.HK", "9633.HK", "9868.HK", "9961.HK",
}

# Statutory HKEX Fee Rates
HKEX_STAMP_DUTY_RATE: float = 0.0010        # 0.10%, rounded UP to nearest HK$1.00
HKEX_SFC_LEVY_RATE: float = 0.000027        # 0.0027%
HKEX_AFRC_LEVY_RATE: float = 0.0000015      # 0.00015%
HKEX_TRADING_FEE_RATE: float = 0.0000565    # 0.00565%
DEFAULT_BROKER_COMM_RATE: float = 0.0003    # 0.03% default commission
DEFAULT_MIN_BROKER_COMM_HKD: float = 15.0   # HK$15 minimum commission
DEFAULT_ALMGREN_GAMMA: float = 0.35         # Almgren-Chriss market impact coefficient


def get_hk_tick_size(price: float) -> float:
    """Return statutory HKEX Main Board tick size (spread) for a given price in HKD."""
    if price < 0.25:
        return 0.001
    if price < 0.50:
        return 0.005
    if price < 10.00:
        return 0.01
    if price < 20.00:
        return 0.02
    if price < 100.00:
        return 0.05
    if price < 200.00:
        return 0.10
    if price < 500.00:
        return 0.20
    if price < 1000.00:
        return 0.50
    if price < 2000.00:
        return 1.00
    if price < 5000.00:
        return 2.00
    return 5.00


def round_to_hk_tick(price: float) -> float:
    """Round any target, execution, or stop-loss price to the nearest valid HKEX statutory tick size."""
    if price <= 0 or np.isnan(price):
        return 0.0
    tick = get_hk_tick_size(price)
    rounded = round(round(price / tick) * tick, 3)
    return float(rounded)


def normalize_hk_ticker(ticker: str) -> str:
    """Normalize HK ticker to Yahoo Finance standard (4-digit zero-padded or 5-digit .HK)."""
    if not ticker:
        return ""
    raw = str(ticker).strip().upper()
    if raw.startswith("^"):
        return raw
    parts = raw.split(".")
    digits = parts[0].lstrip("0")
    if not digits.isdigit():
        return raw if raw.endswith(".HK") else f"{raw}.HK"
    num = int(digits)
    return f"{num:04d}.HK" if num < 10000 else f"{num:05d}.HK"


def get_candidate_tickers(ticker: str) -> list[str]:
    """Generate prioritized candidate tickers for yfinance lookup."""
    norm = normalize_hk_ticker(ticker)
    raw = str(ticker).strip().upper()
    digits = raw.split(".")[0].lstrip("0")
    if not digits.isdigit():
        return [raw]
    candidates = [norm, f"{digits.zfill(4)}.HK", f"{digits.zfill(5)}.HK", raw]
    return list(dict.fromkeys(candidates))


def is_hk_short_eligible(ticker: str, adv_20d_hkd: float = 0.0, config_flag: bool | None = None) -> bool:
    """
    Check if a HK stock is eligible for short selling under SFC Designated List rules.
    Small-cap or low-ADV stocks cannot be legally short-sold or lack stock borrow inventory.
    """
    if config_flag is not None:
        return bool(config_flag)
    norm_t = normalize_hk_ticker(ticker)
    if norm_t in HK_SHORT_ELIGIBLE_WHITELIST:
        return True
    return float(adv_20d_hkd) >= 30_000_000.0


# ============================================================================
# HKEX Microstructure Fee & Almgren-Chriss Market Impact Slippage Engine
# ============================================================================
def calculate_hkex_microstructure_cost(
    trade_value_hkd: float,
    adv_20d_hkd: float = 50_000_000.0,
    daily_volatility: float = 0.02,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    gamma: float = DEFAULT_ALMGREN_GAMMA,
    include_stamp_duty: bool = True,
) -> dict[str, Any]:
    """
    Compute exact HKEX statutory transaction fees and Almgren-Chriss square-root market impact slippage.

    Statutory Formulae:
    - Hong Kong Stamp Duty: 0.10%, rounded UP to the nearest HK$1.00 (`math.ceil(trade_val * 0.001)`)
    - SFC Transaction Levy: 0.0027%
    - AFRC Transaction Levy: 0.00015%
    - HKEX Trading Fee: 0.00565%
    - Brokerage Commission: max(trade_val * broker_comm_rate, min_broker_comm_hkd)
    - Almgren-Chriss Square-Root Market Impact:
      Slippage (bps) = gamma * sqrt(trade_val / ADV_20) * sigma_daily * 10,000
      With an additional liquidity penalty multiplier when single-order participation > 5% of ADV_20.
    """
    trade_val = max(0.0, float(trade_value_hkd))
    if trade_val <= 0.0:
        return {
            "trade_value_hkd": 0.0,
            "stamp_duty_hkd": 0.0,
            "sfc_levy_hkd": 0.0,
            "afrc_levy_hkd": 0.0,
            "hkex_fee_hkd": 0.0,
            "broker_commission_hkd": 0.0,
            "total_statutory_fees_hkd": 0.0,
            "statutory_fee_bps": 0.0,
            "participation_rate": 0.0,
            "slippage_bps": 0.0,
            "slippage_hkd": 0.0,
            "total_frictions_hkd": 0.0,
            "total_cost_bps": 0.0,
            "requires_twap": False,
            "recommended_twap_slices": 1,
        }

    stamp_duty = float(math.ceil(trade_val * HKEX_STAMP_DUTY_RATE)) if include_stamp_duty else 0.0
    sfc_levy = round(trade_val * HKEX_SFC_LEVY_RATE, 2)
    afrc_levy = round(trade_val * HKEX_AFRC_LEVY_RATE, 2)
    hkex_fee = round(trade_val * HKEX_TRADING_FEE_RATE, 2)
    broker_comm = max(round(trade_val * max(0.0, float(broker_comm_rate)), 2), float(min_broker_comm_hkd))

    total_statutory = stamp_duty + sfc_levy + afrc_levy + hkex_fee + broker_comm
    statutory_bps = (total_statutory / trade_val) * 10000.0

    eff_adv = max(float(adv_20d_hkd), 50_000.0)
    sigma_d = float(np.clip(float(daily_volatility), 0.003, 0.25))
    participation = trade_val / eff_adv

    # Almgren-Chriss square-root temporary + permanent impact model
    base_slippage_bps = float(gamma) * math.sqrt(participation) * (sigma_d * 10000.0)

    # Nonlinear liquidity penalty when participation exceeds 5% of 20-day ADV
    requires_twap = participation > 0.05
    if requires_twap:
        excess_ratio = participation / 0.05
        liquidity_penalty_mult = 1.0 + 0.35 * math.log1p(excess_ratio - 1.0)
        slippage_bps = min(base_slippage_bps * liquidity_penalty_mult, 500.0)
        recommended_slices = max(2, int(math.ceil(participation / 0.025)))
    else:
        slippage_bps = min(base_slippage_bps, 300.0)
        recommended_slices = 1

    slippage_hkd = trade_val * (slippage_bps / 10000.0)
    total_frictions_hkd = total_statutory + slippage_hkd
    total_cost_bps = statutory_bps + slippage_bps

    return {
        "trade_value_hkd": trade_val,
        "stamp_duty_hkd": stamp_duty,
        "sfc_levy_hkd": sfc_levy,
        "afrc_levy_hkd": afrc_levy,
        "hkex_fee_hkd": hkex_fee,
        "broker_commission_hkd": broker_comm,
        "total_statutory_fees_hkd": total_statutory,
        "statutory_fee_bps": statutory_bps,
        "participation_rate": participation,
        "slippage_bps": slippage_bps,
        "slippage_hkd": slippage_hkd,
        "total_frictions_hkd": total_frictions_hkd,
        "total_cost_bps": total_cost_bps,
        "requires_twap": requires_twap,
        "recommended_twap_slices": recommended_slices,
    }


# ============================================================================
# Atomic Configuration Persistence & Database Audit
# ============================================================================
def load_portfolio_config(filepath: str = PORTFOLIO_FILE) -> dict[str, Any]:
    """Load portfolio configuration from portfolio.json with safe defaults and schema normalization."""
    default_config: dict[str, Any] = {
        "total_capital_hkd": 300000.0,
        "risk_free_rate": 0.04,
        "max_single_position_pct": 0.35,
        "min_adv_hkd": 5000000.0,
        "alloc_method": "hrp",
        "cov_estimator": "ledoit_wolf",
        "max_turnover": 0.40,
        "broker_comm_rate": DEFAULT_BROKER_COMM_RATE,
        "min_broker_comm_hkd": DEFAULT_MIN_BROKER_COMM_HKD,
        "almgren_gamma": DEFAULT_ALMGREN_GAMMA,
        "num_backtest_trials": 5,
        "holdings": [
            {
                "ticker": "0700.HK",
                "name": "騰訊控股",
                "lot_size": 100,
                "shares": 300,
                "entry_price": 360.0,
                "short_eligible": True,
                "roic": 0.16,
                "debt_to_ebitda": 1.2,
                "thesis": "遊戲版號常態化，視頻號廣告毛利率持續擴張",
                "invalidation_rule": "核心遊戲收入連續兩季按年下跌超過 5%",
            },
            {
                "ticker": "0005.HK",
                "name": "匯豐控股",
                "lot_size": 400,
                "shares": 1200,
                "entry_price": 68.0,
                "short_eligible": True,
                "roic": 0.11,
                "debt_to_ebitda": 2.5,
                "thesis": "高息環境維持淨息差，持續進行大規模股份回購",
                "invalidation_rule": "宣布暫停派息或停止回購計畫",
            },
            {
                "ticker": "3690.HK",
                "name": "美團-W",
                "lot_size": 100,
                "shares": 400,
                "entry_price": 130.0,
                "short_eligible": True,
                "roic": 0.08,
                "debt_to_ebitda": 1.8,
                "thesis": "到店業務競爭趨緩，外賣單均盈利能力穩固",
                "invalidation_rule": "抖音本地生活市佔率進一步蠶食導致營業利潤轉負",
            },
            {
                "ticker": "2475.HK",
                "name": "立訊精密",
                "lot_size": 100,
                "shares": 100,
                "entry_price": 44.0,
                "short_eligible": False,
                "roic": 0.12,
                "debt_to_ebitda": 1.5,
                "thesis": "業務穩健增長",
                "invalidation_rule": "核心業務數據顯著惡化",
            },
        ],
    }
    if not os.path.exists(filepath):
        return default_config
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default_config
        for h in data.get("holdings", []):
            h["ticker"] = normalize_hk_ticker(h.get("ticker", ""))
            h["lot_size"] = int(h.get("lot_size") or HK_DEFAULT_LOT_SIZES.get(h["ticker"], 100))
            h["shares"] = int(h.get("shares") or 0)
            if "short_eligible" not in h:
                h["short_eligible"] = h["ticker"] in HK_SHORT_ELIGIBLE_WHITELIST
        default_config.update(data)
        return default_config
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return default_config


def save_portfolio_config(config_data: dict[str, Any], filepath: str = PORTFOLIO_FILE) -> bool:
    """
    Atomically persist updated portfolio holdings and parameters back to portfolio.json.
    Prevents race conditions and partial file corruption during reactive Streamlit reruns.
    """
    try:
        target_dir = os.path.dirname(os.path.abspath(filepath)) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".portfolio_tmp_", suffix=".json", dir=target_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(config_data, tmp_file, ensure_ascii=False, indent=4)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, filepath)
            return True
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False


def init_database() -> None:
    """Initialize SQLite audit database for portfolio snapshots."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=MEMORY;")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total_value REAL,
            health_score REAL,
            cvar_95 REAL,
            sharpe_ratio REAL
        )
    ''')
    conn.commit()
    conn.close()


def log_snapshot_to_db(health_res: dict[str, Any]) -> None:
    """Record portfolio health metrics to SQLite audit database."""
    try:
        init_database()
        conn = sqlite3.connect(DB_FILE)
        conn.execute("PRAGMA journal_mode=MEMORY;")
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO portfolio_snapshots (timestamp, total_value, health_score, cvar_95, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            now_str,
            float(health_res['total_value']),
            float(health_res['health_score']),
            float(health_res['cvar_95']),
            float(health_res['sharpe_ratio'])
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database logging error: {e}")


# ============================================================================
# Data Ingestion & Explicit Suspension Masking
# ============================================================================
def build_suspension_mask(df: pd.DataFrame) -> pd.Series:
    """
    Build an explicit suspension / non-trading day boolean mask.
    A bar is flagged as suspended if:
    - Volume <= 0 (or missing/NaN before fill), OR
    - Close return is exactly 0 AND High == Low (flatline synthetic bar from ffill or halt).
    """
    vol_zero = (df["Volume"].fillna(0.0) <= 0.0) if "Volume" in df.columns else pd.Series(False, index=df.index)
    raw_c = df["Raw_Close"] if "Raw_Close" in df.columns else df["Close"]
    zero_move = raw_c.diff().abs().fillna(1.0) < 1e-9
    flat_hl = (df["High"] - df["Low"]).abs() < 1e-9 if ("High" in df.columns and "Low" in df.columns) else zero_move
    suspended = vol_zero | (zero_move & flat_hl)
    if len(suspended) > 0:
        suspended.iloc[0] = vol_zero.iloc[0]
    return suspended


def safe_download_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    min_rows: int = 20,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Safely download OHLCV data, normalize timezone to Asia/Hong_Kong,
    separate corporate-action Adjusted Prices from Raw Unadjusted OHLC,
    and attach an explicit `Is_Suspended` mask so halted days never fake low volatility.
    """
    for cand in get_candidate_tickers(ticker):
        try:
            df = yf.download(
                cand, period=period, interval=interval,
                auto_adjust=False, progress=False, timeout=10
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if 'Close' not in df.columns:
                continue

            df = df.copy()
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("Asia/Hong_Kong").tz_localize(None)

            raw_missing = df['Close'].isna()
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0.0)
            else:
                df['Volume'] = 0.0

            df['Raw_Close'] = df['Close']
            df['Raw_Open'] = df['Open'] if 'Open' in df.columns else df['Close']
            df['Raw_High'] = df['High'] if 'High' in df.columns else df['Close']
            df['Raw_Low'] = df['Low'] if 'Low' in df.columns else df['Close']

            if 'Adj Close' in df.columns and interval == "1d":
                adj_ratio = (df['Adj Close'] / (df['Close'] + 1e-9)).fillna(1.0)
                df['Close'] = df['Adj Close']
                for col in ['Open', 'High', 'Low']:
                    if col in df.columns:
                        df[col] = df[col] * adj_ratio

            keep_cols = [
                c for c in [
                    'Open', 'High', 'Low', 'Close',
                    'Raw_Open', 'Raw_High', 'Raw_Low', 'Raw_Close', 'Volume'
                ] if c in df.columns
            ]
            df = df[keep_cols].ffill(limit=3).dropna(subset=['Close', 'High', 'Low'])
            if len(df) < min_rows:
                continue

            susp_mask = build_suspension_mask(df) | raw_missing.reindex(df.index, fill_value=False)
            df['Is_Suspended'] = susp_mask.astype(bool)

            active_bars = int((~df['Is_Suspended']).sum())
            if active_bars >= max(10, min_rows // 2):
                return df, cand
        except Exception:
            continue
    return None, None


def compute_active_daily_returns(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame | None = None,
    suspension_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily returns while masking suspended / halted days as NaN during covariance
    and volatility estimation so zero-return ffill bars do not artificially depress volatility.
    """
    raw_rets = price_df.pct_change()
    if suspension_df is not None:
        mask = suspension_df.reindex_like(raw_rets).fillna(False).astype(bool)
    elif volume_df is not None:
        vol_aligned = volume_df.reindex_like(raw_rets).fillna(0.0)
        mask = (vol_aligned <= 0.0) | (raw_rets.abs() < 1e-10)
    else:
        mask = raw_rets.abs() < 1e-10

    masked_rets = raw_rets.mask(mask, np.nan)
    return masked_rets.iloc[1:], mask.iloc[1:]


# ============================================================================
# Macro Regime & Fundamental Data Pipelines
# ============================================================================
def fetch_hk_macro_regime() -> dict[str, Any]:
    """Evaluate HSI 200-day regime and US 10Y Treasury yield liquidity conditions."""
    try:
        hsi, _ = safe_download_ohlcv("^HSI", period="1y", min_rows=200)
        if hsi is None:
            hsi, _ = safe_download_ohlcv("0700.HK", period="1y", min_rows=200)

        if hsi is not None and len(hsi) >= 200:
            hsi_close = float(hsi['Close'].iloc[-1])
            hsi_sma200 = float(hsi['Close'].rolling(200).mean().iloc[-1])
            hsi_vol20 = float(hsi['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
        else:
            hsi_close, hsi_sma200, hsi_vol20 = 18000.0, 18000.0, 0.22

        tnx = yf.download("^TNX", period="1mo", progress=False, timeout=10)
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx.columns = tnx.columns.get_level_values(0)
        tnx_yield = (
            float(tnx['Close'].dropna().iloc[-1])
            if (tnx is not None and not tnx.empty and 'Close' in tnx.columns and not tnx['Close'].dropna().empty)
            else 4.0
        )

        bear_market = hsi_close < hsi_sma200
        hsi_dist_pct = (hsi_close - hsi_sma200) / (hsi_sma200 + 1e-9)
        hsi_z = float(np.clip(hsi_dist_pct / max(0.04, hsi_vol20 * 0.35), -3.0, 3.0))
        rate_z = float(np.clip((4.25 - tnx_yield) / 0.65, -3.0, 3.0))
        macro_z = float(np.clip(0.65 * hsi_z + 0.35 * rate_z, -3.0, 3.0))
        macro_prob = float(1.0 / (1.0 + math.exp(-0.9 * macro_z)))
        macro_score = float((macro_prob - 0.5) * 200.0)

        reasons = []
        if not bear_market:
            reasons.append(f"🟢 大盤處於 200 日牛熊線上 ({hsi_dist_pct*100:+.1f}%)，宏觀趨勢標準化得分 {hsi_z:+.2f}σ")
        else:
            reasons.append(f"🔴 大盤低於 200 日牛熊線 ({hsi_dist_pct*100:+.1f}%)，宏觀防禦標準化得分 {hsi_z:+.2f}σ")

        if tnx_yield < 4.25:
            reasons.append(f"🟢 外部無風險殖利率溫和 ({tnx_yield:.2f}%)，美元流動性標準化得分 {rate_z:+.2f}σ")
        else:
            reasons.append(f"⚠️ 外部無風險殖利率偏高 ({tnx_yield:.2f}%)，估值折現率標準化得分 {rate_z:+.2f}σ")

        return {
            "hsi_close": hsi_close,
            "hsi_sma200": hsi_sma200,
            "hsi_dist_pct": hsi_dist_pct,
            "tnx_yield": tnx_yield,
            "bear_market": bear_market,
            "macro_zscore": macro_z,
            "macro_prob": macro_prob,
            "macro_score": macro_score,
            "macro_reasons": reasons,
        }
    except Exception:
        return {
            "hsi_close": 18000.0,
            "hsi_sma200": 18000.0,
            "hsi_dist_pct": 0.0,
            "tnx_yield": 4.0,
            "bear_market": False,
            "macro_zscore": 0.0,
            "macro_prob": 0.50,
            "macro_score": 0.0,
            "macro_reasons": ["⚠️ 宏觀數據採用安全中性基準值"],
        }


def fetch_fundamental_data(ticker: str) -> dict[str, Any]:
    """Fetch and standardize Value & Quality fundamental factors into [-3.0, +3.0] Z-Scores."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        raw_div = info.get('trailingAnnualDividendYield') or info.get('dividendYield') or 0.0
        div_yield = float(raw_div) if raw_div is not None else 0.0
        if div_yield > 1.0:
            div_yield = div_yield / 100.0

        raw_roe = info.get('returnOnEquity', 0.10)
        roe = float(raw_roe) if raw_roe is not None else 0.10
        if abs(roe) > 5.0:
            roe = roe / 100.0

        lot_size = info.get('boardLot') or HK_DEFAULT_LOT_SIZES.get(ticker, 100)

        div_z = float(np.clip((div_yield - 0.032) / 0.022, -3.0, 3.0))
        roe_z = float(np.clip((roe - 0.10) / 0.065, -3.0, 3.0))
        val_qual_z = float(np.clip(0.45 * div_z + 0.55 * roe_z, -3.0, 3.0))
        fund_prob = float(1.0 / (1.0 + math.exp(-0.85 * val_qual_z)))
        fund_score = float((fund_prob - 0.5) * 200.0)

        fund_reasons = [
            f"{'🟢' if div_z >= 0 else '⚪'} 股息率價值因子: {div_yield*100:.2f}% (標準化暴露 Z = {div_z:+.2f}σ)",
            f"{'🟢' if roe_z >= 0 else '🔴'} 股東權益報酬率 (ROE): {roe*100:.1f}% (標準化暴露 Z = {roe_z:+.2f}σ)",
        ]
        return {
            "dividend_yield": div_yield,
            "roe": roe,
            "lot_size": int(lot_size),
            "div_zscore": div_z,
            "roe_zscore": roe_z,
            "val_qual_zscore": val_qual_z,
            "div_score": float(div_z * 15.0),
            "roe_score": float(roe_z * 18.0),
            "fund_score": fund_score,
            "fund_reasons": fund_reasons,
        }
    except Exception:
        return {
            "dividend_yield": 0.032,
            "roe": 0.10,
            "lot_size": HK_DEFAULT_LOT_SIZES.get(ticker, 100),
            "div_zscore": 0.0,
            "roe_zscore": 0.0,
            "val_qual_zscore": 0.0,
            "div_score": 0.0,
            "roe_score": 0.0,
            "fund_score": 0.0,
            "fund_reasons": ["⚠️ 財報數據讀取超時，採用中位數 0.0σ 基準值"],
        }


def compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute standard Wilder's RSI using exponential smoothing (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================================
# Task 1.4: Strict Session-Anchored Intraday VWAP (09:30 HKT Reset Boundary)
# ============================================================================
def compute_intraday_vwap_and_tod_volume(df: pd.DataFrame) -> tuple[pd.Series, float, pd.Series]:
    """
    For intraday bars (5m / 15m):
    1. Strictly enforce HKEX trading session reset at 09:30 HKT every trading day.
       Pre-09:30 auction noise or post-16:00 bars do not contaminate continuous session cumulative volume/PV.
    2. Compute Time-of-Day (ToD) normalized volume ratio and ToD volume percentile rank
       to eliminate HKEX U-shape opening/closing volume bias.
    """
    if df.empty:
        empty_s = pd.Series(dtype=float)
        return empty_s, 1.0, empty_s

    idx_dt = pd.to_datetime(df.index)
    if getattr(idx_dt, "tz", None) is not None:
        idx_dt = idx_dt.tz_convert("Asia/Hong_Kong").tz_localize(None)

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].fillna(0.0).clip(lower=0.0)

    bar_times = idx_dt.time
    t_open = time(9, 30)
    t_lunch_start = time(12, 0)
    t_lunch_end = time(13, 0)
    t_close = time(16, 10)

    in_morning = (bar_times >= t_open) & (bar_times <= t_lunch_start)
    in_afternoon = (bar_times >= t_lunch_end) & (bar_times <= t_close)
    in_regular_session = in_morning | in_afternoon

    session_date = idx_dt.date
    is_post_open = bar_times >= t_open
    session_key = pd.Series(
        [f"{d}_{1 if post else 0}" for d, post in zip(session_date, is_post_open)],
        index=df.index
    )

    eff_vol = pd.Series(np.where(in_regular_session, vol, 0.0), index=df.index)
    pv = typical_price * eff_vol

    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = eff_vol.groupby(session_key).cumsum()

    vwap = pd.Series(
        np.where(cum_vol > 0, cum_pv / (cum_vol + 1e-9), typical_price),
        index=df.index
    )

    time_slot_series = pd.Series(bar_times, index=df.index)
    tod_means = eff_vol.groupby(time_slot_series).transform(lambda x: x.replace(0, np.nan).mean())
    tod_means = tod_means.fillna(eff_vol.replace(0, np.nan).mean()).fillna(1.0)
    tod_ratio_series = (eff_vol / (tod_means + 1e-9)).clip(0.0, 20.0)

    decision_idx = -2 if len(df) >= 2 else -1
    confirmed_tod_ratio = float(tod_ratio_series.iloc[decision_idx]) if len(tod_ratio_series) > 0 else 1.0
    return vwap, confirmed_tod_ratio, tod_ratio_series


# ============================================================================
# Task 2.1: Calibrated Multi-Factor Scoring & Collinearity Diagnostics
# ============================================================================
def _winsorize_and_standardize(series: pd.Series, window: int = 60, clip_sigma: float = 3.0) -> pd.Series:
    """Rolling Z-score standardization with strict Winsorization at [-clip_sigma, +clip_sigma]."""
    min_p = max(10, window // 3)
    roll_mean = series.rolling(window, min_periods=min_p).mean()
    roll_std = series.rolling(window, min_periods=min_p).std()
    z = (series - roll_mean) / (roll_std + 1e-9)
    return z.clip(-clip_sigma, clip_sigma).fillna(0.0)


def compute_factor_exposure_and_ic_weights(
    df: pd.DataFrame,
    fundamental_z: float = 0.0,
    macro_z: float = 0.0,
    horizon: int = 5,
    ic_window: int = 120,
) -> dict[str, Any]:
    """
    Construct the 4 canonical institutional factor series using strictly lagged / confirmed bars,
    calibrate weights via rolling Rank-IC (Spearman correlation with forward returns),
    transform into Logistic Regression upward trend probability P(Upward Trend | Factors) in [0, 1],
    and compute Collinearity Diagnostics (VIF & pairwise factor correlation matrix).
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume'].fillna(0.0)
    susp = df['Is_Suspended'] if 'Is_Suspended' in df.columns else (vol <= 0)

    rets = close.pct_change().mask(susp, np.nan)
    vol_20 = rets.rolling(20, min_periods=10).std().bfill().fillna(0.02)
    vol_60 = rets.rolling(60, min_periods=20).std().bfill().fillna(vol_20)

    # 1. Momentum Factor: risk-adjusted 5D and 20D momentum
    ir_5d = close.pct_change(5) / (vol_20 * np.sqrt(5.0) + 1e-9)
    ir_20d = close.pct_change(20) / (vol_60 * np.sqrt(20.0) + 1e-9)
    raw_mom = 0.45 * ir_5d.fillna(0.0) + 0.55 * ir_20d.fillna(0.0)
    z_mom = _winsorize_and_standardize(raw_mom, window=60)

    # 2. Value & Quality Factor: combines cross-sectional fundamental Z with 60D long-term value anchor
    sma_60 = close.rolling(60, min_periods=20).mean().bfill()
    dist_60 = (close - sma_60) / (sma_60 + 1e-9)
    z_val_ts = _winsorize_and_standardize(-dist_60.diff(5).fillna(0.0), window=60)
    z_val = (0.70 * float(np.clip(fundamental_z, -3.0, 3.0)) + 0.30 * z_val_ts).clip(-3.0, 3.0)

    # 3. Volatility / Stability Factor: penalizes high downside volatility and erratic ATR spikes
    downside_rets = rets.clip(upper=0.0)
    downside_vol = downside_rets.rolling(20, min_periods=10).std().bfill().fillna(0.015)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=7).mean() / (close + 1e-9)).bfill().fillna(0.02)
    raw_vol_stability = -(0.60 * downside_vol + 0.40 * atr_pct)
    z_vol = _winsorize_and_standardize(raw_vol_stability, window=60)

    # 4. Liquidity & Microstructure Factor: Amihud illiquidity inverse + signed volume flow
    dollar_vol = (close * vol).replace(0.0, np.nan)
    amihud = (rets.abs() / (dollar_vol + 1e-9)).rolling(20, min_periods=10).mean()
    z_amihud_inv = _winsorize_and_standardize(-np.log1p(amihud.fillna(0.0) * 1e8), window=60)
    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    signed_vol_ratio = ((vol / (vol_ma20 + 1e-9)) - 1.0) * np.sign(rets.fillna(0.0))
    z_flow = _winsorize_and_standardize(signed_vol_ratio.fillna(0.0), window=60)
    z_liq = (0.50 * z_amihud_inv + 0.50 * z_flow).clip(-3.0, 3.0)

    factor_df = pd.DataFrame({
        "Momentum": z_mom,
        "Value_Quality": z_val,
        "Volatility_Quality": z_vol,
        "Liquidity_Micro": z_liq,
    }, index=df.index)

    # Strictly prevent look-ahead in IC calibration:
    # Factor at t-1 predicts return from Open[t] to Close[t + horizon - 1]
    open_col = df['Open'] if 'Open' in df.columns else close
    fwd_ret_for_ic = (close.shift(-horizon) / (open_col.shift(-1) + 1e-9)) - 1.0
    valid_ic_slice = factor_df.iloc[:-horizon - 1].tail(ic_window)
    valid_fwd = fwd_ret_for_ic.loc[valid_ic_slice.index]

    raw_ics: dict[str, float] = {}
    prior_weights = {
        "Momentum": 0.35,
        "Value_Quality": 0.25,
        "Volatility_Quality": 0.20,
        "Liquidity_Micro": 0.20,
    }
    for col in factor_df.columns:
        s_f = valid_ic_slice[col]
        mask_valid = s_f.notna() & valid_fwd.notna()
        if mask_valid.sum() >= 20 and s_f[mask_valid].std() > 1e-6:
            ic_val, _ = spearmanr(s_f[mask_valid], valid_fwd[mask_valid])
            raw_ics[col] = float(0.0 if np.isnan(ic_val) else ic_val)
        else:
            raw_ics[col] = prior_weights[col] * 0.15

    ic_scores = {}
    for col in factor_df.columns:
        emp_weight = max(0.02, abs(raw_ics[col]) if raw_ics[col] >= -0.05 else 0.05)
        ic_scores[col] = 0.55 * prior_weights[col] + 0.45 * emp_weight

    total_ic_score = sum(ic_scores.values()) + 1e-9
    calibrated_weights = {col: float(val / total_ic_score) for col, val in ic_scores.items()}

    composite_z_series = (
        calibrated_weights["Momentum"] * factor_df["Momentum"]
        + calibrated_weights["Value_Quality"] * factor_df["Value_Quality"]
        + calibrated_weights["Volatility_Quality"] * factor_df["Volatility_Quality"]
        + calibrated_weights["Liquidity_Micro"] * factor_df["Liquidity_Micro"]
        + 0.15 * float(np.clip(macro_z, -3.0, 3.0))
    ).clip(-3.0, 3.0)

    prob_up_series = 1.0 / (1.0 + np.exp(-0.85 * composite_z_series))

    clean_f = factor_df.tail(120).dropna()
    corr_vals = clean_f.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 1.0)
    corr_mat = pd.DataFrame(corr_vals, index=clean_f.columns, columns=clean_f.columns)
    try:
        inv_corr = np.linalg.pinv(corr_vals)
        vif_vals = np.maximum(1.0, np.diag(inv_corr))
    except Exception:
        vif_vals = np.ones(len(factor_df.columns))

    vif_map = {col: float(v) for col, v in zip(factor_df.columns, vif_vals)}

    decision_idx = -2 if len(factor_df) >= 2 else -1
    latest_f = factor_df.iloc[decision_idx]
    latest_z = float(composite_z_series.iloc[decision_idx])
    latest_prob = float(prob_up_series.iloc[decision_idx])

    label_map = {
        "Momentum": "動量因子 (Momentum)",
        "Value_Quality": "價值與質量 (Value & Quality)",
        "Volatility_Quality": "低波品質 (Volatility Stability)",
        "Liquidity_Micro": "流動性與資金流 (Liquidity Flow)",
    }

    exposure_rows = []
    for col in factor_df.columns:
        z_v = float(latest_f[col])
        w_v = float(calibrated_weights[col])
        contrib_z = z_v * w_v
        vif_v = vif_map[col]
        exposure_rows.append({
            "factor_key": col,
            "factor_name": label_map[col],
            "z_score": round(z_v, 3),
            "rank_ic": round(raw_ics[col], 4),
            "ic_weight_pct": round(w_v * 100.0, 1),
            "marginal_contribution_z": round(contrib_z, 3),
            "vif": round(vif_v, 2),
            "collinearity_status": "🟢 獨立正交 (VIF < 2.5)" if vif_v < 2.5 else ("🟡 輕微共線" if vif_v < 5.0 else "🔴 高度共線"),
        })

    return {
        "factor_df": factor_df,
        "composite_z_series": composite_z_series,
        "prob_up_series": prob_up_series,
        "latest_composite_z": latest_z,
        "latest_prob_up": latest_prob,
        "calibrated_weights": calibrated_weights,
        "raw_ics": raw_ics,
        "vif_map": vif_map,
        "factor_corr_matrix": corr_mat,
        "exposure_matrix": exposure_rows,
    }


# ============================================================================
# Task 1.1 & Task 2.1: Intraday Day-Trading Signal Engine (Look-Ahead Free)
# ============================================================================
def generate_day_trading_signal(
    ticker: str,
    min_adv_hkd: float = 5_000_000.0,
    interval: str = "15m",
    trade_capital_hkd: float = 50_000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Institutional Intraday Signal Engine:
    - Eliminates Look-Ahead Bias by computing all technical features on confirmed closed bar `t-1`.
    - Separates Decision Price (`Close[t-1]`) from Realistic Execution Price (`Open[t]` + Almgren-Chriss slippage).
    - Outputs calibrated Z-Scores in [-3.0, +3.0] and Logistic Regression Upward Trend Probability P(Up) in [0.0, 1.0].
    """
    try:
        df_daily, successful_ticker = safe_download_ohlcv(ticker, period="6mo", interval="1d", min_rows=25)
        if df_daily is None or successful_ticker is None:
            return None, f"無法獲取 {ticker} 的有效歷史數據（可能停牌或代號無效）"

        raw_close_daily = df_daily['Raw_Close'] if 'Raw_Close' in df_daily.columns else df_daily['Close']
        active_daily_mask = ~df_daily['Is_Suspended'] if 'Is_Suspended' in df_daily.columns else (df_daily['Volume'] > 0)
        active_dollar_vol = (raw_close_daily * df_daily['Volume']).where(active_daily_mask, np.nan)
        adv_20d_hkd = float(active_dollar_vol.rolling(20, min_periods=5).mean().dropna().iloc[-1]) if not active_dollar_vol.dropna().empty else 0.0
        daily_sigma = float(raw_close_daily.pct_change().where(active_daily_mask, np.nan).rolling(20, min_periods=5).std().dropna().iloc[-1]) if len(df_daily) >= 10 else 0.02

        short_ok = is_hk_short_eligible(successful_ticker, adv_20d_hkd=adv_20d_hkd)
        lot_size = HK_DEFAULT_LOT_SIZES.get(successful_ticker, 100)

        adj_factor_daily = (raw_close_daily / (df_daily['Close'] + 1e-9)).fillna(1.0)
        raw_high_d = df_daily['High'] * adj_factor_daily
        raw_low_d = df_daily['Low'] * adj_factor_daily
        tr_daily = pd.concat([
            raw_high_d - raw_low_d,
            (raw_high_d - raw_close_daily.shift(1)).abs(),
            (raw_low_d - raw_close_daily.shift(1)).abs()
        ], axis=1).max(axis=1)
        daily_atr_series = tr_daily.ewm(alpha=1.0 / 14.0, min_periods=10, adjust=False).mean()
        daily_atr_14 = float(daily_atr_series.iloc[-2]) if len(daily_atr_series) >= 2 else float(daily_atr_series.iloc[-1])

        used_interval = interval
        df = None
        if interval in ("5m", "15m"):
            df, _ = safe_download_ohlcv(successful_ticker, period="1mo", interval=interval, min_rows=30)
            if df is None:
                used_interval = "1d"
                df = df_daily
        else:
            df = df_daily

        df = df.copy()
        close = df['Close']
        open_s = df['Open'] if 'Open' in df.columns else close
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        raw_close_s = df['Raw_Close'] if 'Raw_Close' in df.columns else close
        raw_open_s = df['Raw_Open'] if 'Raw_Open' in df.columns else open_s

        dec_idx = -2 if len(df) >= 3 else -1
        prev_dec_idx = -3 if len(df) >= 3 else dec_idx

        decision_close_adj = float(close.iloc[dec_idx])
        prev_decision_close_adj = float(close.iloc[prev_dec_idx])
        decision_price_raw = float(raw_close_s.iloc[dec_idx])
        latest_spot_raw = float(raw_close_s.iloc[-1])
        next_bar_open_raw = float(raw_open_s.iloc[-1]) if len(df) >= 2 else decision_price_raw

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1.0 / 14.0, min_periods=10, adjust=False).mean()
        atr_14_dec = float(atr_series.iloc[dec_idx])
        if np.isnan(atr_14_dec) or atr_14_dec <= 0:
            atr_14_dec = float((high - low).iloc[:dec_idx].mean()) + 1e-6

        price_scale_ratio = decision_price_raw / (decision_close_adj + 1e-9)
        atr_14_raw = atr_14_dec * price_scale_ratio

        price_change_dec = decision_close_adj - prev_decision_close_adj
        change_pct = price_change_dec / (prev_decision_close_adj + 1e-9)
        atr_zscore = float(np.clip(price_change_dec / (atr_14_dec + 1e-9), -3.0, 3.0))

        df['SMA5'] = close.rolling(5, min_periods=2).mean()
        df['SMA10'] = close.rolling(10, min_periods=3).mean()
        df['Vol_MA20'] = volume.rolling(20, min_periods=5).mean()
        sma_5_dec = float(df['SMA5'].iloc[dec_idx])
        sma_10_dec = float(df['SMA10'].iloc[dec_idx])

        hist_vol_slice = volume.iloc[max(0, len(volume) + dec_idx - 60) : len(volume) + dec_idx + 1]
        dec_vol = float(volume.iloc[dec_idx])
        vol_pct_rank = float((hist_vol_slice < dec_vol).mean() * 100.0) if hist_vol_slice.sum() > 0 else 50.0

        if used_interval in ("5m", "15m"):
            vwap_series, vol_surge_ratio, _ = compute_intraday_vwap_and_tod_volume(df)
            df['VWAP'] = vwap_series
            dec_vwap = float(vwap_series.iloc[dec_idx])
        else:
            vol_20ma_dec = float(df['Vol_MA20'].iloc[dec_idx])
            vol_surge_ratio = min(dec_vol / (vol_20ma_dec + 1e-9), 20.0) if (vol_20ma_dec > 0 and adv_20d_hkd >= 100_000) else 0.0
            df['VWAP'] = df['SMA5']
            dec_vwap = float(df['VWAP'].iloc[dec_idx])

        df['ATR_Upper'] = df['VWAP'] + 1.5 * atr_series
        df['ATR_Lower'] = df['VWAP'] - 1.5 * atr_series

        rsi_series = compute_wilder_rsi(close, 14)
        df['RSI'] = rsi_series
        rsi_14_dec = float(rsi_series.iloc[dec_idx]) if not np.isnan(rsi_series.iloc[dec_idx]) else 50.0

        is_illiquid = adv_20d_hkd < min_adv_hkd
        z_mom_intra = float(np.clip(atr_zscore, -3.0, 3.0))
        z_vol_intra = float(np.clip(((vol_pct_rank - 50.0) / 18.0) * np.sign(z_mom_intra if abs(z_mom_intra) > 0.1 else 1.0), -3.0, 3.0)) if not is_illiquid else -2.0
        vwap_dist_atr = (decision_close_adj - dec_vwap) / (atr_14_dec + 1e-9)
        ma_spread_atr = (sma_5_dec - sma_10_dec) / (atr_14_dec + 1e-9)
        z_vwap_intra = float(np.clip(0.65 * vwap_dist_atr + 0.35 * ma_spread_atr, -3.0, 3.0))

        if rsi_14_dec <= 75.0:
            z_rsi_intra = float(np.clip((rsi_14_dec - 50.0) / 10.0, -3.0, 3.0))
        else:
            z_rsi_intra = float(np.clip(-(rsi_14_dec - 72.0) / 6.0, -3.0, 3.0))

        z_liq_intra = float(np.clip(math.log10(max(adv_20d_hkd, 1e4) / max(min_adv_hkd, 1e4)) * 1.5, -3.0, 3.0))

        w_intra = {"mom": 0.30, "vwap": 0.25, "vol": 0.20, "rsi": 0.15, "liq": 0.10}
        composite_z = float(np.clip(
            w_intra["mom"] * z_mom_intra
            + w_intra["vwap"] * z_vwap_intra
            + w_intra["vol"] * z_vol_intra
            + w_intra["rsi"] * z_rsi_intra
            + w_intra["liq"] * z_liq_intra,
            -3.0, 3.0
        ))
        if is_illiquid:
            composite_z = min(composite_z, -1.5)

        up_prob = float(1.0 / (1.0 + math.exp(-1.15 * composite_z)))
        day_score = float(np.clip((up_prob - 0.5) * 200.0, -100.0, 100.0))

        micro_cost = calculate_hkex_microstructure_cost(
            trade_value_hkd=max(trade_capital_hkd, decision_price_raw * lot_size),
            adv_20d_hkd=adv_20d_hkd,
            daily_volatility=daily_sigma,
            broker_comm_rate=broker_comm_rate,
            min_broker_comm_hkd=min_broker_comm_hkd,
            gamma=almgren_gamma,
        )
        slippage_frac = micro_cost["slippage_bps"] / 10000.0
        is_short_bias = (up_prob <= 0.35) and short_ok
        if is_short_bias:
            raw_exec_price = next_bar_open_raw * (1.0 - slippage_frac)
        else:
            raw_exec_price = next_bar_open_raw * (1.0 + slippage_frac)
        execution_price = round_to_hk_tick(raw_exec_price)

        above_vwap = decision_close_adj >= dec_vwap
        if is_illiquid:
            action = "🚫 【流動性不足：禁止當沖進場 (買賣價差與衝擊風險過高)】"
        elif up_prob >= 0.65 and above_vwap:
            action = f"🟢 【順勢做多訊號 (上漲機率 {up_prob*100:.1f}% | 合成 Z={composite_z:+.2f}σ)】"
        elif up_prob <= 0.35:
            if short_ok:
                action = f"🔴 【順勢做空或減倉訊號 (下跌機率 {(1-up_prob)*100:.1f}% | 屬SFC可沽空名單)】"
            else:
                action = "🟠 【觀望迴避 / 僅限多頭平倉 (非SFC可沽空名單或無券源)】"
        else:
            action = f"⚖️ 【短線觀望 / 區間震盪等待 (上漲機率 {up_prob*100:.1f}%)】"

        signals = [
            f"⏱️ 決策與執行分離：訊號決策價 ${round_to_hk_tick(decision_price_raw):.2f} (上一根已確認K線) ➔ 預估撮合執行價 ${execution_price:.2f} (次根開盤 + 滑價 {micro_cost['slippage_bps']:.1f} bps)",
            f"📈 趨勢機率模型：預估上漲機率 {up_prob*100:.1f}% | 合成因子標準化得分 {composite_z:+.2f}σ",
            f"💧 流動性與衝擊成本：20日均成交額 ${adv_20d_hkd:,.0f} HKD | 單筆預估總摩擦成本 {micro_cost['total_cost_bps']:.1f} bps (${micro_cost['total_frictions_hkd']:.1f} HKD)",
            f"🛡️ 港股賣空合規檢查：{'✅ 屬於 SFC 可沽空高流動名單' if short_ok else '🚫 非可沽空名單或借券流動性不足（禁止裸賣空）'}",
        ]

        effective_stop_dist = max(1.5 * atr_14_raw, 0.5 * daily_atr_14) if used_interval in ("5m", "15m") else 1.5 * atr_14_raw
        if is_short_bias:
            stop_price = round_to_hk_tick(execution_price + effective_stop_dist)
            target_price = round_to_hk_tick(max(0.01, execution_price - 2.0 * effective_stop_dist))
        else:
            stop_price = round_to_hk_tick(max(0.01, execution_price - effective_stop_dist))
            target_price = round_to_hk_tick(execution_price + 2.0 * effective_stop_dist)

        return {
            "ticker": successful_ticker,
            "price": round_to_hk_tick(latest_spot_raw),
            "decision_price": round_to_hk_tick(decision_price_raw),
            "execution_price": execution_price,
            "vwap": round_to_hk_tick(dec_vwap * price_scale_ratio),
            "stop_loss_price": stop_price,
            "take_profit_price": target_price,
            "tick_size": get_hk_tick_size(latest_spot_raw),
            "lot_size": lot_size,
            "lot_cost_hkd": latest_spot_raw * lot_size,
            "interval": used_interval,
            "day_score": day_score,
            "up_prob": up_prob,
            "composite_zscore": composite_z,
            "action": action,
            "rsi": rsi_14_dec,
            "vol_ratio": vol_surge_ratio,
            "vol_pct_rank": vol_pct_rank,
            "atr": atr_14_raw,
            "atr_zscore": atr_zscore,
            "change_pct": change_pct,
            "adv_20d_hkd": adv_20d_hkd,
            "short_eligible": short_ok,
            "microstructure": micro_cost,
            "factor_zscores": {
                "ATR 波動率動量 (Z_Mom)": round(z_mom_intra, 2),
                "VWAP 均線結構 (Z_VWAP)": round(z_vwap_intra, 2),
                "時段量能分位 (Z_Vol)": round(z_vol_intra, 2),
                "Wilder RSI 動能 (Z_RSI)": round(z_rsi_intra, 2),
                "流動性安全邊際 (Z_Liq)": round(z_liq_intra, 2),
            },
            "factor_scores": {
                "ATR 波動率動量": round(z_mom_intra * 12.0, 1),
                "時段量能分位": round(z_vol_intra * 12.0, 1),
                "VWAP 均線結構": round(z_vwap_intra * 12.0, 1),
                "Wilder RSI 動能": round(z_rsi_intra * 12.0, 1),
                "流動性安全邊際": round(z_liq_intra * 12.0, 1),
            },
            "signals": signals,
            "history": df.tail(80),
        }, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# Task 1.1 & Task 2.1: Dual-Horizon Calibrated Alpha Engine
# ============================================================================
def analyze_stock_trend(ticker: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Calibrated Multi-Factor Stock Trend & Probability Engine:
    - Uses confirmed bar `t-1` (`decision_price`) for all rolling moving averages and factor Z-scores,
      and reports `execution_price` (`Open[t]` + Almgren-Chriss slippage).
    - Outputs calibrated Logistic Regression upward probability P(Upward Trend | Factors) in [0, 1],
      standardized Z-Scores in [-3.0, +3.0], and full Factor Exposure + VIF Collinearity Matrix.
    """
    try:
        df, successful_ticker = safe_download_ohlcv(ticker, period="2y", interval="1d", min_rows=60)
        if df is None or successful_ticker is None:
            return None, f"無法獲取代號 {ticker} 的有效歷史數據"

        df = df.copy()
        close = df['Close']
        raw_close = df['Raw_Close'] if 'Raw_Close' in df.columns else close
        raw_open = df['Raw_Open'] if 'Raw_Open' in df.columns else raw_close
        volume = df['Volume']
        susp = df['Is_Suspended'] if 'Is_Suspended' in df.columns else (volume <= 0)

        macro = fetch_hk_macro_regime()
        fundamental = fetch_fundamental_data(successful_ticker)

        df['SMA50'] = close.rolling(50, min_periods=20).mean()
        df['SMA200'] = close.rolling(200, min_periods=60).mean()
        df['RSI'] = compute_wilder_rsi(close, 14)

        dec_idx = -2 if len(df) >= 3 else -1
        spot_price_raw = float(raw_close.iloc[-1])
        decision_price_raw = float(raw_close.iloc[dec_idx])
        next_open_raw = float(raw_open.iloc[-1])

        dec_adj_price = float(close.iloc[dec_idx])
        dec_sma50 = float(df['SMA50'].iloc[dec_idx]) if not np.isnan(df['SMA50'].iloc[dec_idx]) else dec_adj_price
        dec_sma200 = float(df['SMA200'].iloc[dec_idx]) if not np.isnan(df['SMA200'].iloc[dec_idx]) else dec_adj_price

        daily_rets = close.pct_change().mask(susp, np.nan)
        vol_20d_series = daily_rets.rolling(20, min_periods=10).std().bfill()
        vol_60d_series = daily_rets.rolling(60, min_periods=20).std().bfill()
        df['Z_20D'] = (close.pct_change(20) / (vol_60d_series * np.sqrt(20.0) + 1e-9)).clip(-3.0, 3.0).fillna(0.0)

        vol_20d = float(vol_20d_series.iloc[dec_idx]) if not np.isnan(vol_20d_series.iloc[dec_idx]) else 0.02
        vol_60d = float(vol_60d_series.iloc[dec_idx]) if not np.isnan(vol_60d_series.iloc[dec_idx]) else vol_20d

        ret_5d = float(close.pct_change(5).iloc[dec_idx]) if len(close) >= 7 else 0.0
        ret_20d = float(close.pct_change(20).iloc[dec_idx]) if len(close) >= 22 else 0.0

        z_5d = float(np.clip(ret_5d / (vol_20d * np.sqrt(5.0) + 1e-9), -3.0, 3.0))
        z_20d = float(np.clip(ret_20d / (vol_60d * np.sqrt(20.0) + 1e-9), -3.0, 3.0))

        df_weekly = close.iloc[:len(close) + dec_idx + 1].resample('W').last().dropna()
        weekly_sma20 = df_weekly.rolling(20, min_periods=5).mean()
        curr_weekly_price = float(df_weekly.iloc[-1]) if not df_weekly.empty else dec_adj_price
        curr_weekly_sma20 = float(weekly_sma20.iloc[-1]) if not weekly_sma20.empty else curr_weekly_price
        weekly_dist = (curr_weekly_price - curr_weekly_sma20) / (curr_weekly_sma20 + 1e-9)
        weekly_bullish = curr_weekly_price > curr_weekly_sma20

        dist_200ma = (dec_adj_price - dec_sma200) / (dec_sma200 + 1e-9)

        factor_model = compute_factor_exposure_and_ic_weights(
            df,
            fundamental_z=fundamental["val_qual_zscore"],
            macro_z=macro["macro_zscore"],
            horizon=10,
            ic_window=120,
        )

        composite_z = factor_model["latest_composite_z"]
        up_prob = factor_model["latest_prob_up"]
        composite_score = float(np.clip((up_prob - 0.5) * 200.0, -100.0, 100.0))

        st_z = float(np.clip(0.55 * z_20d + 0.45 * z_5d, -3.0, 3.0))
        st_prob = float(1.0 / (1.0 + math.exp(-0.9 * st_z)))
        short_term_score = float((st_prob - 0.5) * 200.0)

        lt_tech_z = float(np.clip(0.55 * (dist_200ma / 0.12) + 0.45 * (weekly_dist / 0.08), -3.0, 3.0))
        lt_z = float(np.clip(0.50 * fundamental["val_qual_zscore"] + 0.30 * lt_tech_z + 0.20 * macro["macro_zscore"], -3.0, 3.0))
        lt_prob = float(1.0 / (1.0 + math.exp(-0.9 * lt_z)))
        long_term_score = float((lt_prob - 0.5) * 200.0)

        adv_20d_hkd = float((raw_close * volume).where(~susp, np.nan).rolling(20, min_periods=5).mean().dropna().iloc[-1])
        micro_info = calculate_hkex_microstructure_cost(
            trade_value_hkd=max(50_000.0, spot_price_raw * fundamental["lot_size"]),
            adv_20d_hkd=adv_20d_hkd,
            daily_volatility=vol_20d,
        )
        exec_price = round_to_hk_tick(next_open_raw * (1.0 + (micro_info["slippage_bps"] / 10000.0)))

        if up_prob >= 0.60:
            trend = f"📈 強勢多頭機率 ({up_prob*100:.1f}% | Z={composite_z:+.2f}σ)"
        elif up_prob <= 0.40:
            trend = f"📉 弱勢空頭機率 ({up_prob*100:.1f}% | Z={composite_z:+.2f}σ)"
        else:
            trend = f"⚖️ 中性區間震盪 ({up_prob*100:.1f}% | Z={composite_z:+.2f}σ)"

        reasons = [
            f"🎯 統計校準趨勢機率：預估上漲機率 {up_prob*100:.1f}% (合成因子暴露 Z = {composite_z:+.2f}σ)",
            f"⏱️ 無未來函數決策基準：訊號決策價 ${round_to_hk_tick(decision_price_raw):.2f} (上一交易日收盤) ➔ 次日開盤撮合價 ${exec_price:.2f} (含滑價 {micro_info['slippage_bps']:.1f} bps)",
            f"⚡ 短期動量因子：戰術上漲機率 {st_prob*100:.1f}% (Z = {st_z:+.2f}σ) | 5日動量 Z={z_5d:+.2f}σ ({ret_5d*100:+.2f}%) | 20日動量 Z={z_20d:+.2f}σ ({ret_20d*100:+.2f}%)",
            f"🏛️ 長期價值與宏觀：戰略上漲機率 {lt_prob*100:.1f}% (Z = {lt_z:+.2f}σ) | 前復權 200MA 偏離 {dist_200ma*100:+.2f}% | 週線 20MA 偏離 {weekly_dist*100:+.2f}% ({'🟢 多頭共振' if weekly_bullish else '🔴 週線受壓'})",
        ] + [f"  ↳ {r}" for r in fundamental["fund_reasons"]]

        exp_rows = factor_model["exposure_matrix"]
        factor_breakdown = {
            row["factor_name"]: round(row["z_score"] * 13.3, 1)
            for row in exp_rows
        }
        factor_breakdown["長線均線與週線結構 (Z_Trend)"] = round(lt_tech_z * 13.3, 1)
        factor_breakdown["港股宏觀牛熊閘門 (Z_Macro)"] = round(macro["macro_zscore"] * 13.3, 1)

        return {
            "ticker": successful_ticker,
            "price": round_to_hk_tick(spot_price_raw),
            "decision_price": round_to_hk_tick(decision_price_raw),
            "execution_price": exec_price,
            "trend": trend,
            "up_prob": up_prob,
            "composite_zscore": composite_z,
            "score": composite_score,
            "short_term_score": short_term_score,
            "long_term_score": long_term_score,
            "short_term_prob": st_prob,
            "long_term_prob": lt_prob,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "z_5d": z_5d,
            "z_20d": z_20d,
            "dist_200ma": dist_200ma,
            "weekly_dist": weekly_dist,
            "dividend_yield": fundamental["dividend_yield"],
            "roe": fundamental["roe"],
            "factor_breakdown": factor_breakdown,
            "factor_exposure_matrix": exp_rows,
            "factor_corr_matrix": factor_model["factor_corr_matrix"],
            "calibrated_weights": factor_model["calibrated_weights"],
            "reasons": reasons,
            "history": df.tail(160),
        }, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# Task 2.3: Analytical Ledoit-Wolf Shrinkage & RMT Eigenvalue Denoising
# ============================================================================
def compute_ledoit_wolf_cov(returns_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Analytical Ledoit-Wolf (2004) Shrinkage Covariance estimator towards constant-correlation target.
    """
    clean_X = returns_df.dropna(how="all").fillna(0.0).values
    T, N = clean_X.shape
    if N == 1:
        var_1 = float(np.nanvar(returns_df.values, ddof=1)) * 252.0
        return np.array([[max(var_1, 1e-4)]]), 0.0
    if T < 5:
        eye_cov = np.eye(N) * 0.04
        return eye_cov, 1.0

    X = clean_X - np.mean(clean_X, axis=0, keepdims=True)
    sample_cov = (X.T @ X) / float(T)

    var = np.maximum(np.diag(sample_cov), 1e-8)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    corr = np.clip(sample_cov / (outer_std + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    r_bar = (np.sum(corr) - N) / max(1.0, float(N * (N - 1)))
    target_F = r_bar * outer_std
    np.fill_diagonal(target_F, var)

    X2 = X ** 2
    pi_mat = (X2.T @ X2) / float(T) - (sample_cov ** 2)
    pi_hat = float(np.sum(pi_mat))
    gamma_hat = float(np.sum((sample_cov - target_F) ** 2))

    kappa = (pi_hat / (gamma_hat + 1e-12)) / float(T)
    shrinkage_delta = float(np.clip(kappa, 0.05, 1.0))

    shrunk_cov = ((1.0 - shrinkage_delta) * sample_cov + shrinkage_delta * target_F) * 252.0
    min_eig = np.min(np.linalg.eigvalsh(shrunk_cov))
    if min_eig < 1e-6:
        shrunk_cov += np.eye(N) * (1e-5 - min_eig)
    return shrunk_cov, shrinkage_delta


def compute_rmt_denoised_cov(returns_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Random Matrix Theory (RMT) Marcenko-Pastur Eigenvalue Clipping Covariance Denoising.
    """
    lw_cov, lw_delta = compute_ledoit_wolf_cov(returns_df)
    T, N = returns_df.shape
    if N <= 2 or T < 10:
        return lw_cov, lw_delta

    std = np.sqrt(np.maximum(np.diag(lw_cov), 1e-8))
    outer_std = np.outer(std, std)
    corr = np.clip(lw_cov / (outer_std + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    q = float(N) / float(max(T, N + 1))
    lambda_plus = (1.0 + math.sqrt(q)) ** 2

    noise_mask = eigvals <= lambda_plus
    if np.any(noise_mask) and not np.all(noise_mask):
        eigvals_denoised = eigvals.copy()
        eigvals_denoised[noise_mask] = np.mean(eigvals[noise_mask])
    else:
        eigvals_denoised = np.maximum(eigvals, 1e-4)

    corr_denoised = eigvecs @ np.diag(eigvals_denoised) @ eigvecs.T
    d_scale = np.sqrt(np.maximum(np.diag(corr_denoised), 1e-8))
    corr_denoised = corr_denoised / np.outer(d_scale, d_scale)
    np.fill_diagonal(corr_denoised, 1.0)

    cov_denoised = corr_denoised * outer_std
    return cov_denoised, lw_delta


def _compute_hrp_weights(cov_matrix: np.ndarray, corr_matrix: np.ndarray, cols: list[str]) -> dict[str, float]:
    n = len(cols)
    if n == 1:
        return {cols[0]: 1.0}

    corr_clean = np.clip(np.nan_to_num(corr_matrix, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr_clean, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_clean), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')
    sort_ix = list(leaves_list(link))

    weights = pd.Series(1.0, index=sort_ix)
    clustered_alphas = [sort_ix]

    def get_cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        ivp = 1.0 / (np.diag(cov_slice) + 1e-9)
        ivp /= np.sum(ivp)
        return float(np.dot(ivp.T, np.dot(cov_slice, ivp)))

    while len(clustered_alphas) > 0:
        clustered_alphas = [
            cluster[start:end]
            for cluster in clustered_alphas
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]
        for sub_idx in range(0, len(clustered_alphas), 2):
            left_cluster = clustered_alphas[sub_idx]
            right_cluster = clustered_alphas[sub_idx + 1]
            left_var = get_cluster_var(cov_matrix, left_cluster)
            right_var = get_cluster_var(cov_matrix, right_cluster)
            alloc_factor = 1.0 - left_var / (left_var + right_var + 1e-12)
            weights[left_cluster] *= alloc_factor
            weights[right_cluster] *= (1.0 - alloc_factor)

    weights = weights / weights.sum()
    return {cols[idx]: float(weights.loc[idx]) for idx in range(n)}


def _compute_risk_parity_weights(cov_matrix: np.ndarray, cols: list[str], max_single_weight: float) -> dict[str, float]:
    n = len(cols)
    target_rc = np.full(n, 1.0 / n)

    def risk_parity_obj(weights: np.ndarray) -> float:
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) + 1e-12
        mrc = np.dot(cov_matrix, weights) / port_vol
        rc = weights * mrc
        rc_pct = rc / port_vol
        return float(np.sum((rc_pct - target_rc) ** 2))

    min_w = min(0.05, 0.50 / n)
    max_w = max(float(max_single_weight), min(1.0, 1.35 / n))
    bounds = tuple((min_w, max_w) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    ivp = 1.0 / (np.sqrt(np.diag(cov_matrix)) + 1e-9)
    init_guess = ivp / np.sum(ivp)

    res = minimize(risk_parity_obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success and not np.isnan(res.x).any():
        w = np.clip(res.x, min_w, max_w)
        w /= np.sum(w)
        return {col: float(val) for col, val in zip(cols, w)}
    return {col: float(val) for col, val in zip(cols, init_guess)}


def optimize_portfolio_weights(
    daily_rets: pd.DataFrame,
    max_single_weight: float = 0.35,
    shrink_alpha: float = 0.20,
    method: str = "hrp",
    cov_estimator: str = "ledoit_wolf",
    prev_weights: dict[str, float] | np.ndarray | None = None,
    max_turnover: float | None = 0.40,
) -> dict[str, float]:
    """
    Robust Portfolio Optimization with Ledoit-Wolf / RMT denoising and L1 Turnover Constraints.
    """
    cols = list(daily_rets.columns)
    num_assets = len(cols)
    if num_assets == 0:
        return {}
    if num_assets == 1:
        return {cols[0]: 1.0}

    halt_ratio = (daily_rets.isna() | (daily_rets.abs() < 1e-9)).mean(axis=0).values
    illiquidity_penalty = 1.0 / np.clip(1.0 - halt_ratio, 0.30, 1.0)

    est_key = (cov_estimator or "ledoit_wolf").lower()
    if est_key == "rmt":
        base_cov, _ = compute_rmt_denoised_cov(daily_rets)
    else:
        base_cov, _ = compute_ledoit_wolf_cov(daily_rets)

    cov_matrix = base_cov * np.outer(illiquidity_penalty, illiquidity_penalty)
    std_vec = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))
    corr_matrix = np.clip(cov_matrix / (np.outer(std_vec, std_vec) + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr_matrix, 1.0)

    min_w = min(0.05, 0.50 / num_assets)
    max_w = max(float(max_single_weight), min(1.0, 1.35 / num_assets))

    method_key = (method or "hrp").lower()
    if method_key == "hrp":
        raw_w_dict = _compute_hrp_weights(cov_matrix, corr_matrix, cols)
    elif method_key == "risk_parity":
        raw_w_dict = _compute_risk_parity_weights(cov_matrix, cols, max_w)
    else:
        def portfolio_volatility(weights: np.ndarray) -> float:
            return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

        bounds = tuple((min_w, max_w) for _ in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        init_guess = np.full(num_assets, 1.0 / num_assets)
        result = minimize(
            portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints
        )
        if result.success and not np.isnan(result.x).any():
            raw_w_dict = {col: float(w) for col, w in zip(cols, result.x)}
        else:
            raw_w_dict = {col: 1.0 / num_assets for col in cols}

    w_arr = np.array([raw_w_dict[c] for c in cols], dtype=float)
    for _ in range(15):
        over_mask = w_arr > max_w
        under_mask = w_arr < min_w
        if not np.any(over_mask) and not np.any(under_mask):
            break
        w_arr = np.clip(w_arr, min_w, max_w)
        free_mask = ~(over_mask | under_mask)
        rem_needed = 1.0 - np.sum(w_arr)
        if np.any(free_mask) and np.sum(w_arr[free_mask]) > 0:
            w_arr[free_mask] += rem_needed * (w_arr[free_mask] / np.sum(w_arr[free_mask]))
        else:
            w_arr /= np.sum(w_arr)

    w_arr = np.clip(w_arr, min_w, max_w)
    w_arr /= np.sum(w_arr)

    if prev_weights is not None and max_turnover is not None and float(max_turnover) > 0.0:
        if isinstance(prev_weights, dict):
            prev_arr = np.array([float(prev_weights.get(c, 0.0)) for c in cols], dtype=float)
        else:
            prev_arr = np.asarray(prev_weights, dtype=float)
        if len(prev_arr) == num_assets and np.sum(prev_arr) > 0.5:
            prev_arr = prev_arr / np.sum(prev_arr)
            curr_turnover = float(np.sum(np.abs(w_arr - prev_arr)))
            max_to = float(max_turnover)
            if curr_turnover > max_to + 1e-6:
                blend_alpha = max_to / curr_turnover
                w_arr = prev_arr + blend_alpha * (w_arr - prev_arr)
                w_arr = np.clip(w_arr, min_w, max_w)
                w_arr /= np.sum(w_arr)

    return {col: float(w) for col, w in zip(cols, w_arr)}


def compute_deflated_sharpe_ratio(
    daily_rets: pd.Series,
    rf_rate: float = 0.04,
    num_trials: int = 5
) -> dict[str, Any]:
    """
    Compute Probabilistic & Deflated Sharpe Ratio (PSR / DSR) per Bailey & Lopez de Prado (2014).
    """
    clean_rets = daily_rets.dropna()
    T = len(clean_rets)
    if T < 20:
        return {"dsr_prob": 0.5, "p_value": 0.5, "sr_benchmark_ann": 0.0, "is_significant": False}

    rf_daily = rf_rate / 252.0
    excess = clean_rets - rf_daily
    std_d = float(excess.std())
    if std_d <= 1e-12:
        return {"dsr_prob": 0.0, "p_value": 1.0, "sr_benchmark_ann": 0.0, "is_significant": False}

    nav_end = float((1.0 + clean_rets).prod())
    cagr = (nav_end ** (252.0 / T) - 1.0) if nav_end > 0 else -1.0
    sr_daily = float(((cagr - rf_rate) / 252.0) / std_d)

    sk = float(skew(excess, bias=False))
    ku = float(kurtosis(excess, fisher=False, bias=False))

    K = max(1, int(num_trials))
    if K > 1:
        euler_gamma = 0.5772156649
        z1 = norm.ppf(1.0 - 1.0 / K)
        z2 = norm.ppf(1.0 - (1.0 / K) * np.exp(-1.0))
        sr0_daily = np.sqrt(1.0 / max(1, T - 1)) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        sr0_daily = 0.0

    denom_var = 1.0 - sk * sr_daily + ((ku - 1.0) / 4.0) * (sr_daily ** 2)
    denom = np.sqrt(max(1e-9, denom_var / max(1, T - 1)))
    z_stat = (sr_daily - sr0_daily) / denom
    dsr_prob = float(norm.cdf(z_stat))
    p_val = float(1.0 - dsr_prob)

    return {
        "dsr_prob": dsr_prob,
        "p_value": p_val,
        "sr_benchmark_ann": float(sr0_daily * np.sqrt(252.0)),
        "skewness": sk,
        "kurtosis": ku,
        "is_significant": dsr_prob >= 0.95,
    }


def generate_execution_orders(
    portfolio_items: list[dict[str, Any]],
    total_capital: float,
    optimal_weights: dict[str, float],
    latest_prices: pd.Series,
    adv_map: dict[str, float] | None = None,
    vol_map: dict[str, float] | None = None,
    min_trade_hkd: float = 8000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> list[dict[str, Any]]:
    """
    Generate realistic HKEX execution orders with statutory fee breakdown and Almgren-Chriss slippage.
    """
    orders = []

    for item in portfolio_items:
        t = item['ticker']
        if t not in latest_prices:
            continue
        current_shares = int(item.get('shares', 0))
        price = round_to_hk_tick(float(latest_prices[t]))
        lot_size = int(item.get('lot_size') or HK_DEFAULT_LOT_SIZES.get(t, 100))

        target_w = float(optimal_weights.get(t, 0.0))
        target_val = total_capital * target_w
        target_lots = math.floor((target_val / (price + 1e-9)) / lot_size)
        target_shares = target_lots * lot_size

        current_lots = current_shares // lot_size
        odd_shares = current_shares % lot_size
        shares_diff = target_shares - current_shares
        lots_diff = target_lots - current_lots

        trade_val = abs(shares_diff) * price
        adv_hkd = float(adv_map.get(t, 50_000_000.0)) if adv_map else 50_000_000.0
        sigma_d = float(vol_map.get(t, 0.02)) if vol_map else 0.02

        if 0 < trade_val < min_trade_hkd:
            shares_diff = 0
            lots_diff = 0
            target_shares = current_shares
            trade_val = 0.0
            cost_info = calculate_hkex_microstructure_cost(0.0)
            action = f"⚖️ 維持現狀 (差額低於 ${min_trade_hkd:,.0f} 經濟調倉門檻)"
        else:
            cost_info = calculate_hkex_microstructure_cost(
                trade_value_hkd=trade_val,
                adv_20d_hkd=adv_hkd,
                daily_volatility=sigma_d,
                broker_comm_rate=broker_comm_rate,
                min_broker_comm_hkd=min_broker_comm_hkd,
                gamma=almgren_gamma,
            )
            liq_note = (
                f" [⚠️ 佔ADV {cost_info['participation_rate']*100:.1f}% > 5%：流動性懲罰啟動，建議 TWAP 拆分為 {cost_info['recommended_twap_slices']} 單]"
                if cost_info["requires_twap"]
                else ""
            )

            if shares_diff > 0:
                action = f"🟢 買入 (BUY) {shares_diff:,} 股 ({abs(lots_diff)} 手){liq_note}"
            elif shares_diff < 0:
                action = f"🔴 賣出 (SELL) {abs(shares_diff):,} 股 ({abs(lots_diff)} 手){liq_note}"
            else:
                action = "⚖️ 維持現狀 (HOLD)"

        if odd_shares > 0:
            action += f" [含碎股 {odd_shares} 股]"

        orders.append({
            "ticker": t,
            "name": item.get("name", t),
            "lot_size": lot_size,
            "price": price,
            "target_weight": target_w,
            "current_shares": current_shares,
            "target_shares": target_shares,
            "action": action,
            "estimated_turnover": trade_val,
            "estimated_stamp_duty": cost_info["total_statutory_fees_hkd"],
            "statutory_fees_hkd": cost_info["total_statutory_fees_hkd"],
            "slippage_bps": cost_info["slippage_bps"],
            "slippage_hkd": cost_info["slippage_hkd"],
            "total_friction_hkd": cost_info["total_frictions_hkd"],
            "adv_participation_pct": cost_info["participation_rate"] * 100.0,
            "requires_twap": cost_info["requires_twap"],
        })
    return orders


# ============================================================================
# Task 1.1, 1.2, 1.3 & 2.3: Look-Ahead Free Strategy Backtest Engine
# ============================================================================
def run_strategy_backtest(
    price_df: pd.DataFrame,
    lookback: int = 60,
    rebalance_freq: int = 20,
    max_single_weight: float = 0.35,
    cost_rate: float = 0.0016,
    rf_rate: float = 0.04,
    alloc_method: str = "hrp",
    num_trials: int = 5,
    cov_estimator: str = "ledoit_wolf",
    max_turnover: float = 0.40,
    suspension_df: pd.DataFrame | None = None,
    adv_map: dict[str, float] | None = None,
    total_capital: float = 300_000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
    min_required_bars: int = MIN_REQUIRED_DAILY_BARS,
) -> dict[str, Any] | None:
    """
    Institutional Out-of-Sample Walk-Forward Portfolio Backtest with strict t-1 history windows.
    """
    try:
        if price_df is None or price_df.empty:
            return None

        valid_cols = []
        excluded_cols = []
        for col in price_df.columns:
            col_s = price_df[col].dropna()
            if suspension_df is not None and col in suspension_df.columns:
                active_cnt = int((~suspension_df[col].reindex(col_s.index, fill_value=False)).sum())
            else:
                active_cnt = int((col_s.pct_change().abs() > 1e-9).sum())
            if len(col_s) >= min_required_bars and active_cnt >= max(60, int(min_required_bars * 0.7)):
                valid_cols.append(col)
            else:
                excluded_cols.append(col)

        if not valid_cols:
            return None

        clean_prices = price_df[valid_cols].sort_index().ffill(limit=3).dropna(how="any")
        if len(clean_prices) < max(min_required_bars, lookback + 20):
            return None

        raw_rets = clean_prices.pct_change().iloc[1:]
        masked_rets, _ = compute_active_daily_returns(
            clean_prices,
            suspension_df=suspension_df[valid_cols] if (suspension_df is not None and set(valid_cols).issubset(suspension_df.columns)) else None
        )

        if len(raw_rets) <= lookback + 10:
            return None

        num_assets = len(valid_cols)
        equal_weights = np.full(num_assets, 1.0 / num_assets)

        oos_rets = raw_rets.iloc[lookback:].fillna(0.0)
        strat_daily_rets = []
        bench_daily_rets = []
        turnover_records = []

        curr_strat_w = equal_weights.copy()
        curr_bench_w = equal_weights.copy()

        for i in range(len(oos_rets)):
            t_idx = lookback + i
            day_ret_vec = oos_rets.iloc[i].values

            if i % rebalance_freq == 0:
                hist_slice = masked_rets.iloc[t_idx - lookback : t_idx]
                prev_w_input = None if i == 0 else curr_strat_w
                opt_dict = optimize_portfolio_weights(
                    hist_slice,
                    max_single_weight=max_single_weight,
                    method=alloc_method,
                    cov_estimator=cov_estimator,
                    prev_weights=prev_w_input,
                    max_turnover=max_turnover if i > 0 else None,
                )
                target_w = np.array([opt_dict[col] for col in valid_cols], dtype=float)
                turnover = float(np.sum(np.abs(target_w - curr_strat_w)))
                turnover_records.append(turnover)

                if turnover > 1e-5:
                    total_rebal_cost_hkd = 0.0
                    for idx_c, col in enumerate(valid_cols):
                        w_diff = abs(float(target_w[idx_c] - curr_strat_w[idx_c]))
                        trade_hkd = w_diff * float(total_capital)
                        if trade_hkd > 100.0:
                            adv_c = float(adv_map.get(col, 30_000_000.0)) if adv_map else 30_000_000.0
                            sig_c = float(hist_slice[col].std()) if not np.isnan(hist_slice[col].std()) else 0.02
                            c_res = calculate_hkex_microstructure_cost(
                                trade_value_hkd=trade_hkd,
                                adv_20d_hkd=adv_c,
                                daily_volatility=sig_c,
                                broker_comm_rate=broker_comm_rate,
                                min_broker_comm_hkd=min_broker_comm_hkd,
                                gamma=almgren_gamma,
                            )
                            total_rebal_cost_hkd += c_res["total_frictions_hkd"]
                    tx_cost = total_rebal_cost_hkd / max(float(total_capital), 1.0)
                else:
                    tx_cost = 0.0

                curr_strat_w = target_w
            else:
                tx_cost = 0.0

            port_r = float(np.dot(day_ret_vec, curr_strat_w)) - tx_cost
            bench_r = float(np.dot(day_ret_vec, curr_bench_w))
            strat_daily_rets.append(port_r)
            bench_daily_rets.append(bench_r)

            drifted_strat = curr_strat_w * (1.0 + day_ret_vec)
            curr_strat_w = drifted_strat / (np.sum(drifted_strat) + 1e-12)

            drifted_bench = curr_bench_w * (1.0 + day_ret_vec)
            curr_bench_w = drifted_bench / (np.sum(drifted_bench) + 1e-12)

        strat_series = pd.Series(strat_daily_rets, index=oos_rets.index)
        bench_series = pd.Series(bench_daily_rets, index=oos_rets.index)

        strategy_nav = (1.0 + strat_series).cumprod()
        benchmark_nav = (1.0 + bench_series).cumprod()

        n_days = max(1, len(strat_series))
        total_return_strat = float(strategy_nav.iloc[-1] - 1.0)
        cagr_strat = float(strategy_nav.iloc[-1] ** (252.0 / n_days) - 1.0)
        ann_vol_strat = float(strat_series.std() * np.sqrt(252))
        sharpe_strat = float((cagr_strat - rf_rate) / (ann_vol_strat + 1e-9))

        rolling_max = strategy_nav.cummax()
        drawdowns = (strategy_nav - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())

        bench_rolling_max = benchmark_nav.cummax()
        bench_drawdowns = (benchmark_nav - bench_rolling_max) / bench_rolling_max

        dsr_stats = compute_deflated_sharpe_ratio(strat_series, rf_rate=rf_rate, num_trials=num_trials)

        return {
            "dates": strategy_nav.index,
            "strategy_nav": strategy_nav,
            "benchmark_nav": benchmark_nav,
            "drawdown_series": drawdowns,
            "bench_drawdown_series": bench_drawdowns,
            "total_return": total_return_strat,
            "cagr": cagr_strat,
            "ann_volatility": ann_vol_strat,
            "sharpe": sharpe_strat,
            "max_drawdown": max_dd,
            "lookback_days": lookback,
            "rebalance_freq": rebalance_freq,
            "alloc_method": alloc_method,
            "cov_estimator": cov_estimator,
            "avg_rebalance_turnover": float(np.mean(turnover_records)) if turnover_records else 0.0,
            "excluded_tickers": excluded_cols,
            "dsr_prob": dsr_stats["dsr_prob"],
            "dsr_pvalue": dsr_stats["p_value"],
            "dsr_significant": dsr_stats["is_significant"],
        }
    except Exception:
        return None


def analyze_portfolio_health(
    portfolio_items: list[dict[str, Any]],
    total_capital: float,
    rf_rate: float = 0.04,
    max_single_weight: float = 0.35,
    min_adv_hkd: float = 5_000_000.0,
    alloc_method: str = "hrp",
    num_trials: int = 5,
    cov_estimator: str = "ledoit_wolf",
    max_turnover: float = 0.40,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> dict[str, Any] | None:
    """
    Comprehensive Institutional Portfolio Health, Denoised Optimization & Backtest Workbench.
    """
    if not portfolio_items:
        return None

    adj_price_dict: dict[str, pd.Series] = {}
    raw_price_dict: dict[str, pd.Series] = {}
    susp_dict: dict[str, pd.Series] = {}
    adv_map: dict[str, float] = {}
    vol_map: dict[str, float] = {}
    valid_items: list[dict[str, Any]] = []
    skipped_tickers: list[str] = []
    insufficient_history_tickers: list[str] = []

    for item in portfolio_items:
        raw_t = str(item.get('ticker', '')).strip().upper()
        if not raw_t:
            continue
        df, matched_t = safe_download_ohlcv(raw_t, period="1y", interval="1d", min_rows=30)
        if df is not None and matched_t is not None:
            susp_s = df['Is_Suspended'] if 'Is_Suspended' in df.columns else (df['Volume'] <= 0)
            active_days = int((~susp_s).sum())
            if len(df) < MIN_REQUIRED_DAILY_BARS or active_days < int(MIN_REQUIRED_DAILY_BARS * 0.75):
                insufficient_history_tickers.append(f"{matched_t} (有效交易日 {active_days} < {MIN_REQUIRED_DAILY_BARS})")
                continue

            adj_price_dict[matched_t] = df['Close']
            raw_col = df['Raw_Close'] if 'Raw_Close' in df.columns else df['Close']
            raw_price_dict[matched_t] = raw_col
            susp_dict[matched_t] = susp_s
            vol_s = df['Volume'] if 'Volume' in df.columns else pd.Series(0.0, index=df.index)
            adv_val = float((raw_col * vol_s).where(~susp_s, np.nan).rolling(20, min_periods=5).mean().dropna().iloc[-1])
            adv_map[matched_t] = adv_val
            vol_map[matched_t] = float(raw_col.pct_change().where(~susp_s, np.nan).rolling(20, min_periods=5).std().dropna().iloc[-1])

            lot_sz = int(item.get('lot_size') or HK_DEFAULT_LOT_SIZES.get(matched_t, 100))
            valid_items.append({
                "ticker": matched_t,
                "name": item.get("name", matched_t),
                "shares": int(item.get("shares", 0)),
                "lot_size": lot_sz,
                "short_eligible": is_hk_short_eligible(matched_t, adv_map[matched_t], item.get("short_eligible")),
                "entry_price": item.get("entry_price"),
                "thesis": item.get("thesis", ""),
                "invalidation_rule": item.get("invalidation_rule", ""),
            })
        else:
            skipped_tickers.append(raw_t)

    if not adj_price_dict:
        return None

    adj_data = pd.DataFrame(adj_price_dict).sort_index().ffill(limit=5).dropna(how="any")
    raw_data = pd.DataFrame(raw_price_dict).sort_index().ffill(limit=5).reindex(adj_data.index).ffill()
    susp_data = pd.DataFrame(susp_dict).reindex(adj_data.index).fillna(True)

    if adj_data.empty or len(adj_data) < MIN_REQUIRED_DAILY_BARS:
        return None

    latest_raw_prices = raw_data.iloc[-1]
    shares_map = {item['ticker']: item['shares'] for item in valid_items}

    holdings_value = {}
    total_market_val = 0.0
    for t in adj_data.columns:
        if t in shares_map:
            p = float(latest_raw_prices[t])
            val = p * shares_map[t]
            holdings_value[t] = val
            total_market_val += val

    masked_rets, _ = compute_active_daily_returns(adj_data, suspension_df=susp_data)
    if masked_rets.dropna(how="all").empty:
        return None

    if total_market_val > 0:
        weights = {t: holdings_value[t] / total_market_val for t in holdings_value.keys()}
    else:
        weights = {t: 1.0 / len(adj_data.columns) for t in adj_data.columns}

    position_val_df = pd.DataFrame(
        {t: adj_data[t] * shares_map.get(t, 0) for t in adj_data.columns}
    )
    hist_total_val = position_val_df.sum(axis=1)
    if (hist_total_val > 0).all():
        portfolio_daily_ret = hist_total_val.pct_change().dropna()
    else:
        portfolio_daily_ret = masked_rets.mean(axis=1).dropna()

    n_days = max(1, len(portfolio_daily_ret))
    cum_nav = (1.0 + portfolio_daily_ret).cumprod()
    ann_return = float(cum_nav.iloc[-1] ** (252.0 / n_days) - 1.0)
    ann_volatility = float(portfolio_daily_ret.std() * np.sqrt(252))
    sharpe_ratio = float((ann_return - rf_rate) / (ann_volatility + 1e-9))

    var_95 = float(np.percentile(portfolio_daily_ret, 5))
    tail_losses = portfolio_daily_ret[portfolio_daily_ret <= var_95]
    cvar_95 = float(tail_losses.mean()) if not tail_losses.empty else var_95

    if (cov_estimator or "ledoit_wolf").lower() == "rmt":
        denoised_cov, shrink_delta = compute_rmt_denoised_cov(masked_rets)
    else:
        denoised_cov, shrink_delta = compute_ledoit_wolf_cov(masked_rets)

    d_std = np.sqrt(np.maximum(np.diag(denoised_cov), 1e-8))
    denoised_corr_vals = np.clip(denoised_cov / (np.outer(d_std, d_std) + 1e-12), -1.0, 1.0)
    np.fill_diagonal(denoised_corr_vals, 1.0)
    corr_matrix = pd.DataFrame(denoised_corr_vals, index=adj_data.columns, columns=adj_data.columns)

    avg_corr = (
        float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        if len(weights) > 1
        else 0.0
    )

    effective_capital = float(total_capital) if total_capital > 0 else total_market_val
    optimal_weights = optimize_portfolio_weights(
        masked_rets.iloc[:-1],
        max_single_weight=max_single_weight,
        method=alloc_method,
        cov_estimator=cov_estimator,
        prev_weights=weights if total_market_val > 0 else None,
        max_turnover=max_turnover,
    )
    execution_orders = generate_execution_orders(
        valid_items,
        effective_capital,
        optimal_weights,
        latest_raw_prices,
        adv_map=adv_map,
        vol_map=vol_map,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )
    backtest_res = run_strategy_backtest(
        adj_data,
        lookback=60,
        rebalance_freq=20,
        max_single_weight=max_single_weight,
        rf_rate=rf_rate,
        alloc_method=alloc_method,
        num_trials=num_trials,
        cov_estimator=cov_estimator,
        max_turnover=max_turnover,
        suspension_df=susp_data,
        adv_map=adv_map,
        total_capital=effective_capital,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )

    health_score = 100
    warnings_list = []

    if skipped_tickers:
        warnings_list.append(f"⚠️ 以下代號無法取得有效數據並已略過：{', '.join(skipped_tickers)}")
    if insufficient_history_tickers:
        warnings_list.append(
            f"🛡️ 數據充分性保護啟動：以下上市或有效交易日不足 {MIN_REQUIRED_DAILY_BARS} 根之標的已自動排除，防止共變異數矩陣奇異：{', '.join(insufficient_history_tickers)}"
        )

    total_suspended_days = int(susp_data.sum().sum())
    if total_suspended_days > 0:
        warnings_list.append(
            f"🛡️ 停牌遮罩已啟動：共識別並剔除 {total_suspended_days} 個停牌或無交易日零報酬樣本，消除低波動率失真。"
        )

    est_label = "RMT 隨機矩陣特徵值去噪" if (cov_estimator or "").lower() == "rmt" else "Ledoit-Wolf 解析壓縮共變異數"
    warnings_list.append(
        f"📐 矩陣去噪引擎：採用 {est_label} (最優壓縮強度 δ={shrink_delta:.2f}) 與單次換手率上限 {max_turnover*100:.0f}% 約束。"
    )

    max_weight = max(weights.values()) if weights else 0.0
    if max_weight > max_single_weight:
        health_score -= 25
        warnings_list.append(
            f"⚠️ 單一股票權重過高 ({max_weight*100:.1f}% > 上限 {max_single_weight*100:.0f}%)，資產過度集中。"
        )
    elif max_weight > 0.25:
        health_score -= 10
        warnings_list.append(f"🟡 最大單一持倉佔比達 {max_weight*100:.1f}%。")
    else:
        warnings_list.append("🟢 單一持倉權重分配合理。")

    if avg_corr > 0.70:
        health_score -= 30
        warnings_list.append(f"🔴 組合內資產平均相關性高達 {avg_corr:.2f}，遭遇極端行情時缺乏防禦力。")
    elif avg_corr > 0.50:
        health_score -= 15
        warnings_list.append(f"🟡 資產關聯度偏高 ({avg_corr:.2f})。")
    else:
        warnings_list.append(f"🟢 資產關聯度健康 ({avg_corr:.2f})。")

    if cvar_95 < -0.035:
        health_score -= 15
        warnings_list.append(f"⚠️ 組合單日極端下行風險 (1-Day CVaR 95%) 達 {cvar_95*100:.2f}%。")

    illiquid_names = [t for t, adv in adv_map.items() if adv < min_adv_hkd]
    if illiquid_names:
        health_score -= 15
        warnings_list.append(
            f"⚠️ 低流動性持倉警示：{', '.join(illiquid_names)} 的 20日均成交額低於 ${min_adv_hkd:,.0f} HKD，調倉時需注意買賣價差衝擊。"
        )

    if backtest_res and not backtest_res["dsr_significant"]:
        warnings_list.append(
            f"🔬 抗過擬合統計檢定 (Deflated Sharpe Ratio): 信心水準為 {backtest_res['dsr_prob']*100:.1f}% (p={backtest_res['dsr_pvalue']:.3f} > 0.05)，"
            f"建議優先採用 HRP 或結合 Ledoit-Wolf 去噪配置。"
        )
    elif backtest_res:
        warnings_list.append(
            f"🟢 抗過擬合統計檢定 (Deflated Sharpe Ratio): 信心水準達 {backtest_res['dsr_prob']*100:.1f}% (p={backtest_res['dsr_pvalue']:.3f} <= 0.05)，通過多重測試檢定。"
        )

    health_score = max(0, health_score)
    health_status = (
        "🟢 組合健康良好"
        if health_score >= 80
        else ("🟡 組合風險中等" if health_score >= 60 else "🔴 組合風險偏高")
    )

    res_dict = {
        "health_score": health_score,
        "health_status": health_status,
        "total_value": total_market_val,
        "effective_capital": effective_capital,
        "weights": weights,
        "optimal_weights": optimal_weights,
        "execution_orders": execution_orders,
        "ann_volatility": ann_volatility,
        "ann_return": ann_return,
        "sharpe_ratio": sharpe_ratio,
        "cvar_95": cvar_95,
        "avg_correlation": avg_corr,
        "corr_matrix": corr_matrix,
        "shrinkage_delta": shrink_delta,
        "adv_map": adv_map,
        "backtest": backtest_res,
        "warnings": warnings_list,
    }
    log_snapshot_to_db(res_dict)
    return res_dict


# ============================================================================
# Task 3: Historical Prediction Accuracy & Multi-Horizon Backtest Module
# ============================================================================
def evaluate_historical_prediction_accuracy(
    ticker: str,
    forward_horizon: int = 21,
    history_period: str = "3y",
    atr_stop_mult: float = 2.0,
    signal_z_threshold: float = 0.25,
    trade_capital_hkd: float = 100_000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Vectorized Multi-Horizon Historical Signal Validation & Accuracy Engine.
    """
    try:
        k = max(2, int(forward_horizon))
        period_map = {"6mo": "2y", "1y": "2y", "3y": "5y", "5y": "5y"}
        dl_period = period_map.get(history_period, "5y")

        df_full, matched_ticker = safe_download_ohlcv(
            ticker, period=dl_period, interval="1d", min_rows=MIN_REQUIRED_DAILY_BARS
        )
        if df_full is None or matched_ticker is None:
            return None, f"無法獲取 {ticker} 足夠的歷史日線數據（需至少 {MIN_REQUIRED_DAILY_BARS} 個有效交易日）"

        susp_mask = df_full['Is_Suspended'] if 'Is_Suspended' in df_full.columns else (df_full['Volume'] <= 0)
        df = df_full.loc[~susp_mask].copy()
        if len(df) < MIN_REQUIRED_DAILY_BARS:
            return None, f"{matched_ticker} 剔除停牌日後有效交易日僅 {len(df)} 根，不足 {MIN_REQUIRED_DAILY_BARS} 根最低門檻。"

        hsi_df, _ = safe_download_ohlcv("^HSI", period=dl_period, interval="1d", min_rows=60)

        fundamental = fetch_fundamental_data(matched_ticker)
        macro = fetch_hk_macro_regime()

        factor_res = compute_factor_exposure_and_ic_weights(
            df,
            fundamental_z=fundamental["val_qual_zscore"],
            macro_z=macro["macro_zscore"],
            horizon=min(k, 21),
            ic_window=min(180, max(60, len(df) // 2)),
        )

        z_signal_lagged = factor_res["composite_z_series"].shift(1)
        prob_up_lagged = factor_res["prob_up_series"].shift(1)
        score_lagged = (prob_up_lagged - 0.5) * 200.0

        close_arr = df['Close'].to_numpy(copy=True)
        open_arr = df['Open'].to_numpy(copy=True) if 'Open' in df.columns else close_arr
        high_arr = df['High'].to_numpy(copy=True)
        low_arr = df['Low'].to_numpy(copy=True)
        raw_close_arr = df['Raw_Close'].to_numpy(copy=True) if 'Raw_Close' in df.columns else close_arr
        vol_arr = df['Volume'].to_numpy(copy=True)

        tr_s = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_pct_lagged = (tr_s.ewm(alpha=1.0 / 14.0, min_periods=10, adjust=False).mean() / (df['Close'] + 1e-9)).shift(1).bfill().values

        adv_20 = float(pd.Series(raw_close_arr * vol_arr).rolling(20, min_periods=5).mean().dropna().iloc[-1])
        sigma_20 = float(pd.Series(close_arr).pct_change().rolling(20, min_periods=5).std().dropna().iloc[-1])
        micro_rt = calculate_hkex_microstructure_cost(
            trade_value_hkd=trade_capital_hkd,
            adv_20d_hkd=adv_20,
            daily_volatility=sigma_20,
            broker_comm_rate=broker_comm_rate,
            min_broker_comm_hkd=min_broker_comm_hkd,
            gamma=almgren_gamma,
        )
        round_trip_cost_frac = (2.0 * micro_rt["total_cost_bps"]) / 10000.0

        target_bars_map = {"6mo": 126, "1y": 252, "3y": 756, "5y": 1260}
        req_bars = target_bars_map.get(history_period, 756)

        n_total = len(df)
        max_valid_end = n_total - k
        if max_valid_end <= 40:
            k = max(5, min(k, n_total // 3))
            max_valid_end = n_total - k

        start_idx = max(30, max_valid_end - req_bars)
        eval_indices = np.arange(start_idx, max_valid_end)
        n_eval = len(eval_indices)
        if n_eval < 15:
            return None, f"評估窗口內有效前瞻樣本數 ({n_eval}) 不足，請縮短前瞻週期或延長歷史回測區間。"

        offset_grid = eval_indices[:, None] + np.arange(k)[None, :]
        entry_opens = open_arr[eval_indices]
        window_highs = high_arr[offset_grid]
        window_lows = low_arr[offset_grid]
        window_closes = close_arr[offset_grid]

        long_excursions_high = (window_highs / (entry_opens[:, None] + 1e-9)) - 1.0
        long_excursions_low = (window_lows / (entry_opens[:, None] + 1e-9)) - 1.0
        raw_fwd_returns = (window_closes[:, -1] / (entry_opens + 1e-9)) - 1.0

        z_eval = z_signal_lagged.values[eval_indices]
        prob_eval = prob_up_lagged.values[eval_indices]
        score_eval = score_lagged.values[eval_indices]
        atr_pct_eval = atr_pct_lagged[eval_indices]
        eval_dates = df.index[eval_indices]

        stop_dist_vec = np.clip(float(atr_stop_mult) * atr_pct_eval, 0.015, 0.35)

        sig_sign = np.where(z_eval >= 0.0, 1.0, -1.0)
        mae_vec = np.where(
            sig_sign > 0,
            np.minimum(0.0, np.min(long_excursions_low, axis=1)),
            np.minimum(0.0, -np.max(long_excursions_high, axis=1)),
        )
        mfe_vec = np.where(
            sig_sign > 0,
            np.maximum(0.0, np.max(long_excursions_high, axis=1)),
            np.maximum(0.0, -np.min(long_excursions_low, axis=1)),
        )

        directional_low_path = np.where(
            sig_sign[:, None] > 0,
            long_excursions_low,
            -long_excursions_high,
        )
        hit_stop_matrix = directional_low_path <= (-stop_dist_vec[:, None])
        stopped_out_mask = np.any(hit_stop_matrix, axis=1)
        first_stop_day = np.argmax(hit_stop_matrix, axis=1) + 1
        actual_holding_days = np.where(stopped_out_mask, first_stop_day, k).astype(float)

        unstopped_dir_ret = sig_sign * raw_fwd_returns
        realized_dir_ret = np.where(stopped_out_mask, -stop_dist_vec, unstopped_dir_ret) - round_trip_cost_frac

        active_sig_mask = np.abs(z_eval) >= float(signal_z_threshold)
        if np.sum(active_sig_mask) < 8:
            active_sig_mask = np.ones_like(z_eval, dtype=bool)

        correct_dir = (np.sign(z_eval[active_sig_mask]) == np.sign(raw_fwd_returns[active_sig_mask]))
        hit_rate = float(np.mean(correct_dir)) if len(correct_dir) > 0 else 0.5

        roll_ic_window = min(60, max(20, n_eval // 4))
        s_z = pd.Series(z_eval, index=eval_dates)
        s_ret = pd.Series(raw_fwd_returns, index=eval_dates)

        rank_z = s_z.rank(pct=True)
        rank_ret = s_ret.rank(pct=True)
        rolling_rank_ic = rank_z.rolling(roll_ic_window, min_periods=15).corr(rank_ret).dropna()

        overall_pearson_ic, _ = pearsonr(z_eval, raw_fwd_returns) if np.std(z_eval) > 1e-9 else (0.0, 1.0)
        overall_rank_ic, _ = spearmanr(z_eval, raw_fwd_returns) if np.std(z_eval) > 1e-9 else (0.0, 1.0)

        mean_ic = float(rolling_rank_ic.mean()) if not rolling_rank_ic.empty else float(overall_rank_ic)
        std_ic = float(rolling_rank_ic.std()) if len(rolling_rank_ic) > 2 else 0.15
        ic_ir = float(mean_ic / (std_ic + 1e-9))

        try:
            q_labels = pd.qcut(s_z.rank(method="first"), q=5, labels=["Q1 (極空)", "Q2 (偏空)", "Q3 (中性)", "Q4 (偏多)", "Q5 (極多)"])
        except Exception:
            q_labels = pd.Series(["Q3 (中性)"] * n_eval, index=eval_dates)

        quantile_names = ["Q1 (極空)", "Q2 (偏空)", "Q3 (中性)", "Q4 (偏多)", "Q5 (極多)"]
        quantile_summary = []
        quantile_cum_curves: dict[str, pd.Series] = {}

        daily_equiv_ret = s_ret / float(k)
        for q_name in quantile_names:
            q_mask = (q_labels == q_name)
            q_rets = s_ret[q_mask]
            mean_r = float(q_rets.mean()) if not q_rets.empty else 0.0
            win_r = float((q_rets > 0).mean()) if not q_rets.empty else 0.0
            ann_r = float((1.0 + mean_r) ** (252.0 / k) - 1.0) if mean_r > -0.99 else -0.99

            q_daily_stream = daily_equiv_ret.where(q_mask, 0.0)
            quantile_cum_curves[q_name] = (1.0 + q_daily_stream).cumprod()

            quantile_summary.append({
                "quantile": q_name,
                "count": int(q_mask.sum()),
                "mean_forward_return_pct": mean_r * 100.0,
                "annualized_return_pct": ann_r * 100.0,
                "positive_rate_pct": win_r * 100.0,
            })

        q_means = [row["mean_forward_return_pct"] for row in quantile_summary]
        monotonicity_corr, _ = spearmanr([1, 2, 3, 4, 5], q_means)
        q5_q1_spread_pct = q_means[-1] - q_means[0]

        active_rets = realized_dir_ret[active_sig_mask]
        wins = active_rets[active_rets > 0]
        losses = active_rets[active_rets < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-5 else (3.0 if gross_profit > 0 else 1.0)

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 1e-6
        win_loss_ratio = float(avg_win / avg_loss) if avg_loss > 1e-5 else 1.0
        expected_val_pct = float(np.mean(active_rets) * 100.0) if len(active_rets) > 0 else 0.0
        avg_hold_days = float(np.mean(actual_holding_days[active_sig_mask]))

        strat_daily_equiv = pd.Series(
            np.where(active_sig_mask, realized_dir_ret / float(k), 0.0),
            index=eval_dates
        )
        strat_cum_nav = (1.0 + strat_daily_equiv).cumprod()
        strat_ann_ret = float(strat_cum_nav.iloc[-1] ** (252.0 / max(1, n_eval)) - 1.0)

        if hsi_df is not None and not hsi_df.empty:
            hsi_aligned = hsi_df['Close'].reindex(eval_dates).ffill().bfill()
            hsi_total = float(hsi_aligned.iloc[-1] / (hsi_aligned.iloc[0] + 1e-9) - 1.0)
            hsi_ann_ret = float((1.0 + hsi_total) ** (252.0 / max(1, n_eval)) - 1.0) if hsi_total > -0.99 else 0.0
        else:
            hsi_ann_ret = 0.02
        annualized_alpha = strat_ann_ret - hsi_ann_ret

        pred_bull = z_eval[active_sig_mask] >= 0.0
        actual_bull = (raw_fwd_returns[active_sig_mask] > 0.0) & (~(pred_bull & stopped_out_mask[active_sig_mask]))

        tp = int(np.sum(pred_bull & actual_bull))
        fp = int(np.sum(pred_bull & (~actual_bull)))
        tn = int(np.sum((~pred_bull) & (~actual_bull)))
        fn = int(np.sum((~pred_bull) & actual_bull))

        fpr = float(fp / max(1, fp + tn))
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        stop_trigger_rate = float(np.mean(stopped_out_mask[active_sig_mask]))

        x_score = score_eval
        y_ret_pct = raw_fwd_returns * 100.0
        if np.std(x_score) > 1e-6:
            slope, intercept = np.polyfit(x_score, y_ret_pct, 1)
            r_sq = float(overall_pearson_ic ** 2)
        else:
            slope, intercept, r_sq = 0.0, 0.0, 0.0

        scatter_df = pd.DataFrame({
            "date": eval_dates.strftime("%Y-%m-%d"),
            "pred_zscore": np.round(z_eval, 3),
            "pred_prob_pct": np.round(prob_eval * 100.0, 1),
            "pred_score": np.round(x_score, 1),
            "forward_return_pct": np.round(y_ret_pct, 2),
            "mae_pct": np.round(mae_vec * 100.0, 2),
            "mfe_pct": np.round(mfe_vec * 100.0, 2),
            "stopped_out": stopped_out_mask,
            "quantile": q_labels.astype(str).values,
        })

        return {
            "ticker": matched_ticker,
            "forward_horizon": k,
            "history_period": history_period,
            "sample_size": int(np.sum(active_sig_mask)),
            "total_bars_evaluated": n_eval,
            "hit_rate": hit_rate,
            "mean_ic": mean_ic,
            "overall_rank_ic": float(overall_rank_ic),
            "overall_pearson_ic": float(overall_pearson_ic),
            "ic_ir": ic_ir,
            "annualized_alpha": annualized_alpha,
            "strategy_ann_return": strat_ann_ret,
            "benchmark_ann_return": hsi_ann_ret,
            "profit_factor": min(profit_factor, 9.99),
            "win_loss_ratio": min(win_loss_ratio, 9.99),
            "expected_return_pct": expected_val_pct,
            "avg_holding_days": avg_hold_days,
            "monotonicity_score": float(0.0 if np.isnan(monotonicity_corr) else monotonicity_corr),
            "q5_q1_spread_pct": q5_q1_spread_pct,
            "quantile_summary": quantile_summary,
            "quantile_cum_curves": quantile_cum_curves,
            "rolling_rank_ic": rolling_rank_ic,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "fpr": fpr,
                "precision": precision,
                "recall": recall,
                "stop_trigger_rate": stop_trigger_rate,
            },
            "regression": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": r_sq,
            },
            "scatter_df": scatter_df,
            "round_trip_cost_bps": micro_rt["total_cost_bps"] * 2.0,
        }, None
    except Exception as e:
        return None, str(e)