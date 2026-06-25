"""
F1 Pit Stop Predictor — Streamlit Web App
==========================================
Ensemble of XGBoost v9 (per-lap) + 2-layer LSTM (5-lap sequence).
Run with:  streamlit run app.py
"""

import os, sys, subprocess, tempfile, math
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# Custom CSS — F1 dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0F1117;
    color: #E6EDF3;
}
.main { background-color: #0F1117; }
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #E8002D;
    margin: 1.2rem 0 0.4rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #30363D;
}

/* ── Validate button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #E8002D, #B80024);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.5rem 1rem;
    transition: all 0.2s;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #FF1E3C, #E8002D);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(232,0,45,0.4);
}

/* ── Metric cards ── */
.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
    margin-bottom: 0.75rem;
}
.metric-card .val {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    font-family: 'Courier New', monospace;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #8B949E;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Recommendation banner ── */
.rec-banner {
    border-radius: 10px;
    padding: 1.2rem 2rem;
    text-align: center;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    margin: 0.75rem 0;
}
.rec-pit  { background: linear-gradient(135deg,#0D2A0F,#163E18); border: 2px solid #3FB950; color: #3FB950; }
.rec-stay { background: linear-gradient(135deg,#2A0D0D,#3E1616); border: 2px solid #E8002D; color: #E8002D; }

/* ── Confidence badge ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.badge-high   { background:#3FB95022; color:#3FB950; border:1px solid #3FB95066; }
.badge-medium { background:#D2992222; color:#D29922; border:1px solid #D2992266; }
.badge-low    { background:#8B949E22; color:#8B949E; border:1px solid #8B949E66; }
.badge-none   { background:#30363D44; color:#666;    border:1px solid #30363D;   }

/* ── Model vote cards ── */
.vote-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.vote-card .model-name { font-size:0.72rem; color:#8B949E; text-transform:uppercase; letter-spacing:0.08em; }
.vote-card .model-prob { font-size:1.5rem; font-weight:800; font-family:'Courier New',monospace; margin-top:0.1rem; }

/* ── Feature table ── */
.feat-table { width:100%; border-collapse:collapse; font-size:0.83rem; }
.feat-table th {
    background:#21262D; color:#8B949E; padding:0.4rem 0.7rem;
    text-align:left; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;
    border-bottom:1px solid #30363D;
}
.feat-table td { padding:0.35rem 0.7rem; border-bottom:1px solid #21262D; color:#C9D1D9; }
.feat-table tr:last-child td { border-bottom:none; }
.feat-table tr:hover td { background:#21262D; }
.feat-derived { color:#D2A8FF; font-style:italic; }

/* ── Section title ── */
.section-title {
    font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:#8B949E;
    margin:1.5rem 0 0.6rem 0;
    padding-bottom:0.3rem;
    border-bottom:1px solid #30363D;
}
/* ── Title bar ── */
.f1-title {
    font-size:1.6rem; font-weight:900; letter-spacing:0.04em;
    background:linear-gradient(90deg,#E8002D,#FF6B6B);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0;
}
.f1-subtitle { color:#8B949E; font-size:0.85rem; margin-top:0.1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
THRESHOLD = 0.65

FEATURE_COLS = [
    "compound","tire_age","tire_age_squared","rolling_avg_time","degradation_rate",
    "gap_ahead","gap_behind","position","race_completion","undercut_threat",
    "circuit_id","pit_loss_time","track_temp","rainfall",
    "ahead_compound","ahead_tire_age","behind_compound","behind_tire_age",
    "expected_stint_length",
]

CIRCUIT_NAMES = {
     0:"Abu Dhabi Grand Prix",   1:"Australian Grand Prix",
     2:"Austrian Grand Prix",    3:"Azerbaijan Grand Prix",
     4:"Bahrain Grand Prix",     5:"Belgian Grand Prix",
     6:"British Grand Prix",     7:"Canadian Grand Prix",
     8:"Dutch Grand Prix",       9:"Emilia Romagna Grand Prix",
    10:"French Grand Prix",     11:"Hungarian Grand Prix",
    12:"Italian Grand Prix",    13:"Japanese Grand Prix",
    14:"Las Vegas Grand Prix",  15:"Mexico City Grand Prix",
    16:"Miami Grand Prix",      17:"Monaco Grand Prix",
    18:"Portuguese Grand Prix", 19:"Qatar Grand Prix",
    20:"Russian Grand Prix",    21:"Saudi Arabian Grand Prix",
    22:"Singapore Grand Prix",  23:"Spanish Grand Prix",
    24:"Styrian Grand Prix",    25:"São Paulo Grand Prix",
    26:"Turkish Grand Prix",    27:"United States Grand Prix",
}

PIT_LOSS_TIME = {
     0: 3.39, 1:14.60, 2: 3.58, 3: 4.40, 4: 3.01, 5: 4.22,
     6:-0.37, 7:18.76, 8: 5.61, 9: 6.35,10:19.78,11: 2.74,
    12: 5.00,13: 4.67,14: 6.39,15: 3.69,16: 5.01,17:22.20,
    18: 3.93,19:13.71,20:12.23,21: 5.09,22: 9.03,23: 3.92,
    24: 4.87,25: 4.62,26: 6.68,27: 1.62,
}

GLOBAL_MEAN_STINT_LENGTH = 23.8
EXPECTED_STINT_LENGTH = {
    ( 0,0):14.2,( 0,1):18.2,( 0,2):25.1, ( 1,0): 4.5,( 1,1): 9.9,( 1,2):37.0,
    ( 2,0):14.6,( 2,1):23.9,( 2,2):28.1, ( 3,0): 8.8,( 3,1):10.1,( 3,2):32.8,
    ( 4,0):14.7,( 4,1):18.6,( 4,2):18.4, ( 5,0):13.8,( 5,1):15.5,( 5,2):12.0,
    ( 6,0):19.4,( 6,1):24.3,( 6,2):24.3, ( 7,0): 6.0,( 7,1):18.4,( 7,2):31.5,
    ( 8,0):21.9,( 8,1):28.0,( 8,2):31.8, ( 9,0):19.5,( 9,1):26.4,
    (10,0):19.0,(10,1):20.5,(10,2):32.0, (11,0): 9.0,(11,1):22.5,(11,2):32.2,
    (12,0): 1.0,(12,1):21.4,(12,2):26.6, (13,0):10.9,(13,1):15.5,(13,2):18.3,
    (14,0):10.0,(14,1):16.0,(14,2):23.3, (15,0): 9.0,(15,1):28.4,(15,2):32.8,
    (16,0): 4.5,(16,1):17.6,(16,2):40.7, (17,0):33.0,(17,1):33.2,(17,2):41.3,
    (18,0):21.6,(18,1):35.8,(18,2):32.8, (19,0):17.1,(19,1):18.8,(19,2):20.4,
    (20,0): 2.0,(20,1):17.4,(20,2):30.4, (21,0): 9.6,(21,1):18.7,(21,2):22.9,
    (22,0):16.5,(22,1):19.9,(22,2):32.5, (23,0):19.2,(23,1):25.2,(23,2):24.6,
    (24,0):20.8,(24,1):29.9,(24,2):36.9, (25,0):22.1,(25,1):25.2,(25,2):24.5,
                (26,1): 5.0,              (27,0): 8.8,(27,1):16.4,(27,2):21.0,
}

COMPOUND_LABELS = {0:"Soft", 1:"Medium", 2:"Hard"}

# ─────────────────────────────────────────────────────────────
# LSTM architecture (must match train_model_lstm.py)
# ─────────────────────────────────────────────────────────────
class PitLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(self.dropout(h_n[-1]))

# ─────────────────────────────────────────────────────────────
# Model loading (cached across reruns)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_lstm_model():
    path = os.path.join(APP_DIR, "pit_model_lstm.pt")
    if not os.path.exists(path):
        return None, None, f"pit_model_lstm.pt not found in {APP_DIR}"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = PitLSTM(len(FEATURE_COLS), 64, 2, 0.3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        scaler = StandardScaler()
        scaler.mean_  = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

# ─────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────
def predict_xgb(feat_array: np.ndarray) -> tuple[float | None, str | None]:
    """
    Run XGBoost in a subprocess to avoid torch+xgb MKL conflict.

    IMPORTANT: pit_model_v9.json was saved from XGBClassifier with
    early_stopping_rounds=30.  It contains 97 total trees but best_iteration=66
    is stored as a Booster attribute.  Booster.predict() without iteration_range
    uses ALL 97 trees and returns ~42% for the Bahrain scenario.  We must use
    iteration_range=(0, best_iteration+1) to match XGBClassifier.predict_proba().
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "X.npy")
        op = os.path.join(tmpdir, "p.npy")
        np.save(fp, feat_array.astype(np.float32))
        script = (
            f"import numpy as np, xgboost as xgb\n"
            f"X  = np.load({fp!r})\n"
            f"b  = xgb.Booster()\n"
            f"b.load_model('pit_model_v9.json')\n"
            f"# Read best_iteration saved by XGBClassifier early stopping\n"
            f"_bi = b.attributes().get('best_iteration')\n"
            f"_n  = int(_bi) + 1 if _bi is not None else b.num_boosted_rounds()\n"
            f"dm = xgb.DMatrix(X.astype(float))\n"
            f"p  = b.predict(dm, iteration_range=(0, _n))\n"
            f"np.save({op!r}, p)\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=APP_DIR,
        )
        if r.returncode != 0:
            return None, r.stderr[-400:] if r.stderr else "Unknown XGBoost error"
        return float(np.load(op)[0]), None



def predict_lstm_fn(feat_1d: np.ndarray, model, scaler) -> float:
    """
    Build a realistic 5-lap degradation sequence and run the LSTM.

    Laps t-4 … t receive increasing degradation offsets (in real seconds,
    applied BEFORE scaling so the scaler sees coherent feature magnitudes):
        t-4: degradation_rate − 2.0 s
        t-3: degradation_rate − 1.5 s
        t-2: degradation_rate − 1.0 s
        t-1: degradation_rate − 0.5 s
        t  : actual current values  (no offset)
    """
    deg_idx = FEATURE_COLS.index("degradation_rate")  # = 4
    offsets = [-2.0, -1.5, -1.0, -0.5, 0.0]

    # Build 5 unscaled rows, varying only degradation_rate
    seq_raw = np.tile(feat_1d, (5, 1))          # (5, 19) — unscaled copy
    for i, off in enumerate(offsets):
        seq_raw[i, deg_idx] = feat_1d[deg_idx] + off

    # Scale all 5 rows together so relative magnitudes are correct
    x_sc = scaler.transform(seq_raw).astype(np.float32)  # (5, 19)
    x_t  = torch.from_numpy(x_sc).unsqueeze(0)           # (1, 5, 19)
    with torch.no_grad():
        return torch.sigmoid(model(x_t)).item()


# ─────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────
def compute_features(inp: dict) -> tuple[dict, float, float, float]:
    rc    = inp["current_lap"] / inp["total_laps"]
    ta2   = inp["tire_age"] ** 2
    bta   = inp["behind_tire_age"]
    if inp["gap_behind"] >= 999.0 or bta < 0:
        uct = 0.0
    else:
        denom = inp["tire_age"] - bta + 1
        uct   = 0.0 if denom <= 0 else inp["gap_behind"] / denom
    esl   = EXPECTED_STINT_LENGTH.get((inp["circuit_id"], inp["compound"]),
                                      GLOBAL_MEAN_STINT_LENGTH)
    feats = {
        "compound":              inp["compound"],
        "tire_age":              inp["tire_age"],
        "tire_age_squared":      ta2,
        "rolling_avg_time":      inp["rolling_avg"],
        "degradation_rate":      inp["degradation"],
        "gap_ahead":             inp["gap_ahead"],
        "gap_behind":            inp["gap_behind"],
        "position":              inp["position"],
        "race_completion":       rc,
        "undercut_threat":       uct,
        "circuit_id":            inp["circuit_id"],
        "pit_loss_time":         PIT_LOSS_TIME[inp["circuit_id"]],
        "track_temp":            inp["track_temp"],
        "rainfall":              inp["rainfall"],
        "ahead_compound":        inp["ahead_compound"],
        "ahead_tire_age":        inp["ahead_tire_age"],
        "behind_compound":       inp["behind_compound"],
        "behind_tire_age":       bta,
        "expected_stint_length": esl,
    }
    return feats, rc, uct, esl


# ─────────────────────────────────────────────────────────────
# SVG Arc Gauge
# ─────────────────────────────────────────────────────────────
def arc_gauge_html(prob: float, threshold: float = THRESHOLD) -> str:
    # ── Geometry: BOTH arcs use the same CX, CY, R, SW, M, LAF, SF ──────
    CX, CY, R = 110, 105, 82   # centre, radius
    SW        = 14              # stroke-width — identical for bg and fill

    sx, sy = CX - R, CY        # start point  (28, 105)  — angle = 180°
    bx, by = CX + R, CY        # bg end point (192, 105) — angle = 0°

    # Fill arc endpoint: angle maps p=0→π, p=1→0
    p_c   = max(0.003, min(0.997, prob))
    angle = math.pi * (1.0 - p_c)
    px    = CX + R * math.cos(angle)
    py    = CY - R * math.sin(angle)
    # Check: p=0.5 → angle=π/2 → (110, 23) — exact top midpoint of bg arc ✓

    # large-arc-flag = 0 ALWAYS (gauge sweeps ≤ 180°, never needs long arc)
    # sweep-flag     = 1 ALWAYS (clockwise for both arcs)
    LAF, SF = 0, 1

    # Threshold tick
    t_a  = math.pi * (1.0 - threshold)
    ti_x = CX + (R - SW * 0.8) * math.cos(t_a)
    ti_y = CY - (R - SW * 0.8) * math.sin(t_a)
    to_x = CX + (R + SW * 0.8) * math.cos(t_a)
    to_y = CY - (R + SW * 0.8) * math.sin(t_a)
    tl_x = CX + (R + SW * 1.9) * math.cos(t_a)
    tl_y = CY - (R + SW * 1.9) * math.sin(t_a)

    bar_col = "#3FB950" if prob >= threshold else "#D29922" if prob >= 0.40 else "#8B949E"
    pct_txt = f"{prob * 100:.1f}%"

    bg_d   = f"M {sx} {sy} A {R} {R} 0 {LAF} {SF} {bx} {by}"
    fill_d = f"M {sx} {sy} A {R} {R} 0 {LAF} {SF} {px:.3f} {py:.3f}"

    return f"""<div style="display:flex;justify-content:center;">
<svg viewBox="0 0 220 130" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:360px;margin:0 auto">
  <defs>
    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <path d="{bg_d}" fill="none" stroke="#2A2A3A" stroke-width="{SW}" stroke-linecap="round"/>
  <path d="{fill_d}" fill="none" stroke="{bar_col}" stroke-width="{SW}" stroke-linecap="round" filter="url(#glow)"/>
  <line x1="{ti_x:.2f}" y1="{ti_y:.2f}" x2="{to_x:.2f}" y2="{to_y:.2f}"
        stroke="white" stroke-width="2.5" stroke-linecap="round" opacity="0.85"/>
  <text x="{tl_x:.1f}" y="{tl_y + 3:.1f}" text-anchor="middle" font-size="8" fill="#999" font-family="monospace">t={threshold}</text>
  <text x="{CX}" y="{CY - 12}" text-anchor="middle" font-size="30" fill="{bar_col}" font-weight="900"
        font-family="'Courier New',monospace" filter="url(#glow)">{pct_txt}</text>
  <text x="{CX}" y="{CY + 7}" text-anchor="middle" font-size="9" fill="#555" letter-spacing="2" font-family="sans-serif">PIT PROBABILITY</text>
  <text x="{sx - 4}" y="{sy + 16}" text-anchor="middle" font-size="8" fill="#444">0%</text>
  <text x="{bx + 4}" y="{by + 16}" text-anchor="middle" font-size="8" fill="#444">100%</text>
</svg></div>
"""


# ─────────────────────────────────────────────────────────────
# Session-state defaults (Bahrain validation scenario)
# ─────────────────────────────────────────────────────────────
# Keys must match ACTUAL widget key= parameters (not just logical names).
# circuit_idx_widget : the selectbox key (stores the int index 0-27)
# _rainfall_radio    : the radio key (stores the selected label string)
BAHRAIN_SCENARIO = dict(
    # ── widget keys that Streamlit reads on rerun ──
    circuit_idx_widget = 4,           # Bahrain = index 4 in circuit_options list
    current_lap        = 14,
    total_laps         = 57,
    position           = 1,
    compound           = 0,           # 0 = Soft
    tire_age           = 14,
    rolling_avg        = 93.5,
    degradation        = 3.5,
    gap_ahead          = 999.0,       # leading
    gap_behind         = 1.8,
    ahead_compound     = -1,          # -1 = Leading / no car ahead
    ahead_tire_age     = -1.0,
    behind_compound    = 0,           # 0 = Soft (Perez)
    behind_tire_age    = 14.0,
    track_temp         = 38,
    _rainfall_radio    = "\u2600\ufe0f  Dry",  # exact radio label string
    # ── logical alias kept for compute_features ──
    circuit_id         = 4,
    rainfall           = 0,
)

def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def _init_defaults():
    _ss("circuit_id",     0)
    _ss("current_lap",    20)
    _ss("total_laps",     57)
    _ss("position",       8)
    _ss("compound",       1)
    _ss("tire_age",       12)
    _ss("rolling_avg",    92.0)
    _ss("degradation",    0.15)
    _ss("gap_ahead",      2.5)
    _ss("gap_behind",     3.0)
    _ss("ahead_compound", 1)
    _ss("ahead_tire_age", 10.0)
    _ss("behind_compound",1)
    _ss("behind_tire_age",8.0)
    _ss("track_temp",     38.0)
    _ss("rainfall",       0)

_init_defaults()

def load_scenario(s: dict):
    for k, v in s.items():
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# Load LSTM model once
# ─────────────────────────────────────────────────────────────
lstm_model, lstm_scaler, lstm_err = load_lstm_model()

# ─────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="f1-title">🏎 F1 PREDICTOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="f1-subtitle">Ensemble · XGBoost v9 + LSTM</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Validate button ──────────────────────────────────────
    if st.button("🏆  Validate: VER · Bahrain 2023 · Lap 14"):
        load_scenario(BAHRAIN_SCENARIO)
        st.rerun()

    # ── Circuit ──────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Circuit</p>', unsafe_allow_html=True)
    circuit_options = list(CIRCUIT_NAMES.values())
    circuit_ids     = list(CIRCUIT_NAMES.keys())
    sel_circuit_idx = st.selectbox(
        "Circuit", options=range(len(circuit_options)),
        format_func=lambda i: circuit_options[i],
        index=st.session_state["circuit_id"],
        key="circuit_idx_widget",
        label_visibility="collapsed",
    )
    st.session_state["circuit_id"] = circuit_ids[sel_circuit_idx]
    cid = st.session_state["circuit_id"]

    # ── Race Context ─────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Race Context</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        current_lap = st.number_input("Current Lap", 1, 100,
                                       value=int(st.session_state["current_lap"]),
                                       key="current_lap")
    with c2:
        total_laps = st.number_input("Total Laps", current_lap, 100,
                                      value=max(int(st.session_state["total_laps"]),
                                                current_lap),
                                      key="total_laps")
    position = st.number_input("Position", 1, 20,
                                value=int(st.session_state["position"]),
                                key="position")

    # ── Tire State ───────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Tire State</p>', unsafe_allow_html=True)
    cmp_options = ["Soft", "Medium", "Hard"]
    compound = st.selectbox("Compound", options=[0, 1, 2],
                             format_func=lambda x: cmp_options[x],
                             index=int(st.session_state["compound"]),
                             key="compound")
    tire_age = st.slider("Tire Age (laps)", 1, 50,
                          value=int(st.session_state["tire_age"]),
                          key="tire_age")

    # ── Pace & Degradation ───────────────────────────────────
    st.markdown('<p class="sidebar-section">Pace & Degradation</p>',
                unsafe_allow_html=True)
    rolling_avg = st.number_input("Rolling Avg Lap Time (s)", 60.0, 180.0,
                                   value=float(st.session_state["rolling_avg"]),
                                   step=0.1, format="%.1f", key="rolling_avg")
    degradation = st.number_input("Degradation Rate (s)", -5.0, 10.0,
                                   value=float(st.session_state["degradation"]),
                                   step=0.01, format="%.2f", key="degradation")

    # ── Gap Information ──────────────────────────────────────
    st.markdown('<p class="sidebar-section">Gap Information</p>',
                unsafe_allow_html=True)
    st.caption("Enter 999 if leading / last place")
    gap_ahead  = st.number_input("Gap Ahead (s)", 0.0, 999.0,
                                  value=float(st.session_state["gap_ahead"]),
                                  step=0.1, format="%.1f", key="gap_ahead")
    gap_behind = st.number_input("Gap Behind (s)", 0.0, 999.0,
                                  value=float(st.session_state["gap_behind"]),
                                  step=0.1, format="%.1f", key="gap_behind")

    # ── Opponent State ───────────────────────────────────────
    st.markdown('<p class="sidebar-section">Opponent State</p>',
                unsafe_allow_html=True)

    ahead_opts  = ["Leading (−1)", "Soft", "Medium", "Hard"]
    ahead_vals  = [-1, 0, 1, 2]
    def _ahead_idx():
        v = st.session_state["ahead_compound"]
        return ahead_vals.index(v) if v in ahead_vals else 0

    ahead_compound = st.selectbox("Car Ahead — Compound",
                                   options=ahead_vals,
                                   format_func=lambda x: ahead_opts[ahead_vals.index(x)],
                                   index=_ahead_idx(), key="ahead_compound")

    if ahead_compound == -1:
        ahead_tire_age = -1.0
        st.caption("Leading — no car ahead")
    else:
        ahead_tire_age = st.number_input("Car Ahead — Tire Age (laps)", 0.0, 60.0,
                                          value=float(max(0.0, st.session_state["ahead_tire_age"])),
                                          step=1.0, format="%.0f", key="ahead_tire_age")

    behind_opts = ["Last Place (−1)", "Soft", "Medium", "Hard"]
    behind_vals = [-1, 0, 1, 2]
    def _behind_idx():
        v = st.session_state["behind_compound"]
        return behind_vals.index(v) if v in behind_vals else 0

    behind_compound = st.selectbox("Car Behind — Compound",
                                    options=behind_vals,
                                    format_func=lambda x: behind_opts[behind_vals.index(x)],
                                    index=_behind_idx(), key="behind_compound")

    if behind_compound == -1:
        behind_tire_age = -1.0
        st.caption("Last place — no car behind")
    else:
        behind_tire_age = st.number_input("Car Behind — Tire Age (laps)", 0.0, 60.0,
                                           value=float(max(0.0, st.session_state["behind_tire_age"])),
                                           step=1.0, format="%.0f", key="behind_tire_age")

    # ── Weather ──────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Weather</p>', unsafe_allow_html=True)
    track_temp = st.slider("Track Temperature (°C)", 10, 65,
                            value=int(st.session_state["track_temp"]),
                            key="track_temp")
    rainfall_label = st.radio("Conditions", ["☀️  Dry", "🌧️  Wet"],
                               index=int(st.session_state["rainfall"]),
                               horizontal=True, key="_rainfall_radio")
    rainfall = 1 if rainfall_label.startswith("🌧") else 0

# ─────────────────────────────────────────────────────────────
# Assemble inputs
# ─────────────────────────────────────────────────────────────
inp = dict(
    circuit_id=cid, current_lap=current_lap, total_laps=total_laps,
    position=position, compound=compound, tire_age=float(tire_age),
    rolling_avg=rolling_avg, degradation=degradation,
    gap_ahead=gap_ahead, gap_behind=gap_behind,
    ahead_compound=ahead_compound,
    ahead_tire_age=ahead_tire_age if ahead_compound != -1 else -1.0,
    behind_compound=behind_compound,
    behind_tire_age=behind_tire_age if behind_compound != -1 else -1.0,
    track_temp=float(track_temp), rainfall=rainfall,
)

feats, race_completion, undercut_threat, expected_sl = compute_features(inp)
feat_vec = np.array([[feats[c] for c in FEATURE_COLS]], dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# ── MAIN PANEL ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

# Title row
col_title, col_circuit = st.columns([3, 2])
with col_title:
    st.markdown('<h1 class="f1-title">F1 Pit Stop Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="f1-subtitle">Ensemble · XGBoost v9 + 2-layer LSTM · 19 features</p>',
                unsafe_allow_html=True)
with col_circuit:
    st.markdown(f"""
    <div class="metric-card" style="text-align:right;padding:0.6rem 1rem;">
        <div class="lbl">Circuit</div>
        <div style="font-size:1.1rem;font-weight:700;color:#E6EDF3;margin-top:0.15rem;">
            {CIRCUIT_NAMES[cid]}
        </div>
        <div class="lbl" style="margin-top:0.2rem;">Pit loss: {PIT_LOSS_TIME[cid]:+.2f}s</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Run inference ──────────────────────────────────────────
p_xgb_val  = None
p_lstm_val = None
xgb_err    = None

with st.spinner("Computing probabilities…"):
    p_xgb_val, xgb_err = predict_xgb(feat_vec)
    if lstm_model is not None and lstm_scaler is not None:
        try:
            p_lstm_val = predict_lstm_fn(feat_vec[0], lstm_model, lstm_scaler)
        except Exception as ex:
            lstm_err = str(ex)

# Ensemble probability
if p_xgb_val is not None and p_lstm_val is not None:
    p_ens    = 0.5 * p_xgb_val + 0.5 * p_lstm_val
    mode     = "ensemble"
elif p_xgb_val is not None:
    p_ens    = p_xgb_val
    mode     = "xgb_only"
elif p_lstm_val is not None:
    p_ens    = p_lstm_val
    mode     = "lstm_only"
else:
    p_ens    = 0.0
    mode     = "error"

pit          = p_ens >= THRESHOLD
pct          = p_ens * 100
stint_pct    = 100 * inp["tire_age"] / expected_sl if expected_sl > 0 else 0


# ── Confidence ──────────────────────────────────────────────
if p_ens >= 0.80:
    conf_label, conf_cls, conf_note = "HIGH",   "high",   "Strong signal — model is confident"
elif p_ens >= THRESHOLD:
    conf_label, conf_cls, conf_note = "MEDIUM", "medium", "Above threshold — lean towards pitting"
elif p_ens >= 0.40:
    conf_label, conf_cls, conf_note = "LOW",    "low",    "Borderline — monitor next 2–3 laps"
else:
    conf_label, conf_cls, conf_note = "NONE",   "none",   "No significant pit signal"

# ─────────────────────────────────────────────────────────────
# Layout: gauge + recommendation | feature table
# ─────────────────────────────────────────────────────────────
col_result, col_table = st.columns([1, 1], gap="large")

with col_result:
    # Gauge
    st.markdown(arc_gauge_html(p_ens), unsafe_allow_html=True)

    # Recommendation banner
    if pit:
        st.markdown(
            '<div class="rec-banner rec-pit">🟢  PIT NOW</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rec-banner rec-stay">🔴  STAY OUT</div>',
            unsafe_allow_html=True,
        )

    # Confidence + note
    conf_center = f"""
    <div style="text-align:center;margin:0.6rem 0;">
        <span class="badge badge-{conf_cls}">{conf_label} CONFIDENCE</span>
        <p style="color:#8B949E;font-size:0.82rem;margin-top:0.4rem;">{conf_note}</p>
    </div>"""
    st.markdown(conf_center, unsafe_allow_html=True)

    # Threshold reminder
    st.markdown(
        f'<p style="text-align:center;color:#555;font-size:0.75rem;">'
        f'Operating threshold: {THRESHOLD} (best-F1 on ensemble test set)</p>',
        unsafe_allow_html=True,
    )

    # ── Individual model votes ─────────────────────────────
    st.markdown('<div class="section-title">Model Votes</div>', unsafe_allow_html=True)
    vc1, vc2, vc3 = st.columns(3)

    def _vote_card(col, name, prob, color):
        with col:
            if prob is None:
                html = f"""<div class="vote-card">
                    <div class="model-name">{name}</div>
                    <div class="model-prob" style="color:#555;font-size:1rem;">N/A</div>
                </div>"""
            else:
                decision = "PIT" if prob >= THRESHOLD else "STAY"
                html = f"""<div class="vote-card">
                    <div class="model-name">{name}</div>
                    <div class="model-prob" style="color:{color};">{prob*100:.1f}%</div>
                    <div style="font-size:0.7rem;color:#555;margin-top:0.15rem;">{decision}</div>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)

    _vote_card(vc1, "XGBoost v9", p_xgb_val, "#D2A8FF")
    _vote_card(vc2, "LSTM",       p_lstm_val, "#FF7B72")
    ens_col = "#3FB950" if pit else "#E8002D"
    _vote_card(vc3, "Ensemble",   p_ens,      ens_col)

    if mode != "ensemble":
        st.warning("⚠️ One model unavailable — showing single-model result.")
    if xgb_err:
        with st.expander("XGBoost error"):
            st.code(xgb_err)
    if lstm_err:
        with st.expander("LSTM error"):
            st.code(lstm_err)

with col_table:
    st.markdown('<div class="section-title">Feature Summary</div>', unsafe_allow_html=True)

    cmp_str  = {0:"Soft",1:"Medium",2:"Hard"}.get(compound, "?")
    acmp_str = {-1:"Leading (−1)",0:"Soft",1:"Medium",2:"Hard"}.get(ahead_compound, "?")
    bcmp_str = {-1:"Last (−1)",  0:"Soft",1:"Medium",2:"Hard"}.get(behind_compound, "?")
    rain_str = "🌧 Wet" if rainfall else "☀ Dry"

    rows_inp = [
        ("compound",        f"{cmp_str} ({compound})",        False),
        ("tire_age",        f"{tire_age} laps",               False),
        ("rolling_avg_time",f"{rolling_avg:.2f} s",           False),
        ("degradation_rate",f"{degradation:+.3f} s",          False),
        ("gap_ahead",       "LEADING" if gap_ahead>=999 else f"{gap_ahead:.2f} s", False),
        ("gap_behind",      "LAST"    if gap_behind>=999 else f"{gap_behind:.2f} s",False),
        ("position",        str(int(position)),                False),
        ("track_temp",      f"{track_temp}°C",                False),
        ("rainfall",        rain_str,                          False),
        ("ahead_compound",  acmp_str,                          False),
        ("ahead_tire_age",  "−1 (leading)" if ahead_compound==-1 else f"{ahead_tire_age:.0f} laps", False),
        ("behind_compound", bcmp_str,                          False),
        ("behind_tire_age", "−1 (last)"    if behind_compound==-1 else f"{behind_tire_age:.0f} laps",False),
    ]

    rows_derived = [
        ("race_completion",      f"{race_completion*100:.1f}%",       True),
        ("tire_age_squared",     f"{inp['tire_age']**2:.0f}",         True),
        ("undercut_threat",      f"{undercut_threat:.4f}",            True),
        ("expected_stint_length",f"{expected_sl:.1f} laps  ({stint_pct:.0f}% through)", True),
        ("pit_loss_time",        f"{PIT_LOSS_TIME[cid]:+.2f} s",      True),
        ("circuit_id",           str(cid),                            True),
    ]

    all_rows = rows_inp + rows_derived
    rows_html = "".join(
        f'<tr><td{"  class=\"feat-derived\"" if d else ""}>{n}</td>'
        f'<td style="text-align:right;font-family:monospace;">{v}</td></tr>'
        for n, v, d in all_rows
    )

    table_html = f"""
    <table class="feat-table">
      <thead><tr><th>Feature</th><th style="text-align:right">Value</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="color:#555;font-size:0.72rem;margin-top:0.5rem;">
    <em style="color:#D2A8FF;">Italic purple</em> = auto-computed from inputs</p>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # Stint progress bar
    st.markdown('<div class="section-title" style="margin-top:1rem;">Stint Progress</div>',
                unsafe_allow_html=True)
    st_pct_clamped = min(100, int(stint_pct))
    bar_col = "#E8002D" if stint_pct > 90 else "#D29922" if stint_pct > 70 else "#3FB950"
    st.markdown(f"""
    <div style="background:#2A2A3A;border-radius:4px;height:10px;overflow:hidden;margin-bottom:0.3rem;">
      <div style="background:{bar_col};width:{st_pct_clamped}%;height:100%;
                  border-radius:4px;transition:width 0.3s;"></div>
    </div>
    <p style="font-size:0.78rem;color:#8B949E;margin:0;">
      Lap {tire_age} of ~{expected_sl:.0f} expected  ({stint_pct:.0f}% through
      typical {cmp_str} stint at {CIRCUIT_NAMES[cid].split(' ')[0]})
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#555;font-size:0.75rem;">'
    'Ensemble: XGBoost v9 (ROC-AUC 0.8363, AP 0.1689) + '
    '2-layer LSTM (ROC-AUC 0.8478, AP 0.1488) · '
    '<strong style="color:#3FB950;">Ensemble: ROC-AUC 0.9289 · AP 0.3523 · 9.6× lift</strong>'
    '</p>',
    unsafe_allow_html=True,
)
