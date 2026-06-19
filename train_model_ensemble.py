"""
Ensemble Predictor — XGBoost v9 + LSTM (equal-weight average)
=============================================================
Reconstructs the exact same 2023 sequence test set used by train_model_lstm.py
(5-lap windows, stratified 80/20, seed=42).

For each test sequence:
  - XGBoost v9 predicts on the FINAL lap of the sequence (raw, unscaled features)
  - LSTM predicts on the full 5-lap sequence (StandardScaler applied)
  - Ensemble = 0.5 × p_xgb + 0.5 × p_lstm

Evaluates ROC-AUC, AP, best F1 threshold; plots PR curve.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score, recall_score, f1_score,
)

# ── Baselines ─────────────────────────────────────────────────────────────────
V9   = {"roc": 0.8363, "ap": 0.1689, "label": "v9  (XGBoost, 19 feat, Optuna)"}
LSTM_B = {"roc": 0.8478, "ap": 0.1488, "label": "LSTM (2-layer, 5-lap seq)"}

# ── Config (must match train_model_lstm.py exactly) ───────────────────────────
SEQ_LEN    = 5
HIDDEN_DIM = 64
N_LAYERS   = 2
DROPOUT    = 0.3
SEED       = 42
XGB_WEIGHT = 0.5
LST_WEIGHT = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}\n")

# ── Feature set (19 features — identical to v9 and LSTM) ─────────────────────
FEATURE_COLS = [
    "compound", "tire_age", "tire_age_squared",
    "rolling_avg_time", "degradation_rate",
    "gap_ahead", "gap_behind", "position",
    "race_completion", "undercut_threat",
    "circuit_id", "pit_loss_time",
    "track_temp", "rainfall",
    "ahead_compound", "ahead_tire_age",
    "behind_compound", "behind_tire_age",
    "expected_stint_length",
]
LABEL_COL = "pitted_next_lap"

# ── Data prep (mirrors v9 / LSTM preprocessing) ───────────────────────────────
raw = pd.read_csv("f1_pit_data_2021_2023.csv")
df  = raw[raw["season"] == 2023].copy()
print(f"Full dataset : {len(raw):,} rows × {raw.shape[1]} cols")
print(f"2023 filter  : {len(df):,} rows\n")

df = df.dropna(subset=["rolling_avg_time", "degradation_rate", "position"])
df["gap_ahead"]  = df["gap_ahead"].fillna(999.0)
df["gap_behind"] = df["gap_behind"].fillna(999.0)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.reset_index(drop=True)

# ── Load scaler params from LSTM checkpoint ───────────────────────────────────
lstm_ckpt  = torch.load("pit_model_lstm.pt", map_location="cpu", weights_only=False)
scaler     = StandardScaler()
scaler.mean_  = lstm_ckpt["scaler_mean"]
scaler.scale_ = lstm_ckpt["scaler_scale"]

# Scale features (for LSTM)
X_raw    = df[FEATURE_COLS].values.astype(float)   # unscaled — XGBoost
X_scaled = scaler.transform(X_raw)                 # scaled   — LSTM
y_all    = df[LABEL_COL].values.astype(float)

df_meta = df[["round", "driver", "lap_number"]].copy()
df_meta["row_idx"] = np.arange(len(df_meta))

# ── Build sequences (identical logic to train_model_lstm.py) ─────────────────
def build_sequences(meta, X_raw, X_scaled, y, seq_len=5):
    """
    Returns:
      seqs_raw    : (N, n_feat)   — final lap raw features for XGBoost
      seqs_scaled : (N, seq_len, n_feat) — full window scaled for LSTM
      labels      : (N,)          — pitted_next_lap of final lap
    """
    seqs_raw, seqs_scaled, labels = [], [], []
    for (rnd, drv), grp in meta.groupby(["round", "driver"]):
        grp  = grp.sort_values("lap_number").reset_index(drop=True)
        laps = grp["lap_number"].values
        idxs = grp["row_idx"].values
        for i in range(len(grp) - seq_len + 1):
            window_laps = laps[i : i + seq_len]
            if np.all(np.diff(window_laps) == 1):
                w = idxs[i : i + seq_len]
                seqs_raw.append(X_raw[w[-1]])          # final lap — unscaled
                seqs_scaled.append(X_scaled[w])        # full window — scaled
                labels.append(y[w[-1]])
    return (np.array(seqs_raw,    dtype=np.float32),
            np.array(seqs_scaled, dtype=np.float32),
            np.array(labels,      dtype=np.float32))

print("Building 5-lap sequences …")
X_r, X_s, y_seq = build_sequences(df_meta, X_raw, X_scaled, y_all, SEQ_LEN)
print(f"Sequences total : {len(y_seq):,}  |  "
      f"pit rate {100*y_seq.mean():.2f}%\n")

# ── Reproduce exact train/test split (same seed as LSTM training) ─────────────
idx_all = np.arange(len(y_seq))
_, idx_te, _, _ = train_test_split(
    idx_all, y_seq, test_size=0.20, random_state=SEED, stratify=y_seq
)

X_te_raw    = X_r[idx_te]          # (n_test, 19)  for XGBoost
X_te_scaled = X_s[idx_te]          # (n_test, 5, 19) for LSTM
y_te        = y_seq[idx_te]

print(f"Test sequences  : {len(y_te):,}  |  pit rate {100*y_te.mean():.2f}%")
print(f"  pit  (1): {int(y_te.sum()):,}")
print(f"  none (0): {int((y_te==0).sum()):,}\n")

# ── Load XGBoost v9 (subprocess to avoid torch+xgb MKL conflict) ─────────────
print("Loading pit_model_v9.json via subprocess …")
import subprocess, tempfile, os, sys

# Save test features to a temp file, run XGBoost in clean subprocess
with tempfile.TemporaryDirectory() as tmpdir:
    feat_path  = os.path.join(tmpdir, "X_te.npy")
    proba_path = os.path.join(tmpdir, "p_xgb.npy")
    np.save(feat_path, X_te_raw.astype(np.float32))

    _script = f"""
import numpy as np, xgboost as xgb, sys
X = np.load({feat_path!r})
b = xgb.Booster()
b.load_model("pit_model_v9.json")
dm = xgb.DMatrix(X.astype(float))
p  = b.predict(dm)
np.save({proba_path!r}, p)
"""
    result = subprocess.run(
        [sys.executable, "-c", _script],
        capture_output=True, text=True,
        cwd="/Users/ahilannayani/Downloads/College/F1 ML Model",
    )
    if result.returncode != 0:
        print("  XGBoost subprocess stderr:", result.stderr[:800])
        raise RuntimeError(f"XGBoost subprocess failed (code {result.returncode})")
    p_xgb = np.load(proba_path)

print(f"  XGBoost  — mean proba: {p_xgb.mean():.4f}  "
      f"max: {p_xgb.max():.4f}  AP: {average_precision_score(y_te, p_xgb):.4f}")
sys.stdout.flush()
print()

# ── Load LSTM ─────────────────────────────────────────────────────────────────
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

print("Loading pit_model_lstm.pt …")
lstm_model = PitLSTM(len(FEATURE_COLS), HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
lstm_model.eval()

# Batch inference
BATCH = 512
p_lstm = []
with torch.no_grad():
    for start in range(0, len(X_te_scaled), BATCH):
        xb = torch.from_numpy(X_te_scaled[start:start+BATCH]).to(DEVICE)
        logits = lstm_model(xb)
        p_lstm.extend(torch.sigmoid(logits).cpu().numpy().flatten())
p_lstm = np.array(p_lstm)
print(f"  LSTM     — mean proba: {p_lstm.mean():.4f}  "
      f"max: {p_lstm.max():.4f}  AP: {average_precision_score(y_te, p_lstm):.4f}\n")

# ── Ensemble (equal-weight average) ──────────────────────────────────────────
p_ens  = XGB_WEIGHT * p_xgb + LST_WEIGHT * p_lstm
y_pred = (p_ens >= 0.50).astype(int)

roc_auc  = roc_auc_score(y_te, p_ens)
ap       = average_precision_score(y_te, p_ens)
baseline = float(y_te.mean())
cm       = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=" * 64)
print(f"  ENSEMBLE  (XGB {XGB_WEIGHT:.0%} + LSTM {LST_WEIGHT:.0%})  —  19 feat, 2023")
print("=" * 64)
print(classification_report(
    y_te, y_pred,
    target_names=["No Pit (0)", "Pit (1)"],
    digits=4,
))
print(f"  ROC-AUC               : {roc_auc:.4f}")
print(f"  Average Precision (AP): {ap:.4f}")
print(f"  Baseline (random)     : {baseline:.4f}")
print(f"  Lift over baseline    : {ap/baseline:.1f}×")
print()
print("  Confusion Matrix")
print("                   Predicted 0   Predicted 1")
print(f"  Actual 0  (no pit)   {cm[0,0]:>6}        {cm[0,1]:>6}")
print(f"  Actual 1  (pit)      {cm[1,0]:>6}        {cm[1,1]:>6}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print("=" * 64)

# ── Version comparison ────────────────────────────────────────────────────────
def _d(new, old):
    diff = new - old
    m = "↑" if diff > 0.0001 else "↓" if diff < -0.0001 else "≈"
    return f"{diff:+.4f} {m}"

print("\n" + "=" * 76)
print("  VERSION COMPARISON")
print("  " + "-" * 72)
print(f"  {'Metric':<30}  {'v9 (XGB)':>10}  {'LSTM':>8}  {'Ensemble':>10}  {'Δ vs best':>12}")
print("  " + "-" * 72)
best_prev_roc = max(V9["roc"], LSTM_B["roc"])
best_prev_ap  = max(V9["ap"],  LSTM_B["ap"])
print(f"  {'ROC-AUC':<30}  {V9['roc']:>10.4f}  {LSTM_B['roc']:>8.4f}  {roc_auc:>10.4f}  {_d(roc_auc, best_prev_roc):>12}")
print(f"  {'Avg Precision (AP)':<30}  {V9['ap']:>10.4f}  {LSTM_B['ap']:>8.4f}  {ap:>10.4f}  {_d(ap, best_prev_ap):>12}")
print(f"  {'Lift over baseline':<30}  {'5.5×':>10}  {'4.0×':>8}  {ap/baseline:.1f}×{'':<7}")
print(f"  {'TP @ t=0.50':<30}  {'68':>10}  {'76':>8}  {tp:>10}  {tp-76:>+12}")
print(f"  {'FN @ t=0.50':<30}  {'53':>10}  {'34':>8}  {fn:>10}  {fn-34:>+12}")
print(f"  {'FP @ t=0.50':<30}  {'501':>10}  {'505':>8}  {fp:>10}  {fp-505:>+12}")
print("  " + "-" * 72)
print()

# ── Threshold table ───────────────────────────────────────────────────────────
prec_pts, rec_pts, thresh_pts = precision_recall_curve(y_te, p_ens)
f1_curve = 2 * prec_pts * rec_pts / np.maximum(prec_pts + rec_pts, 1e-9)
best_idx = int(np.argmax(f1_curve[:-1]))
best_t   = float(thresh_pts[best_idx])
best_f1  = float(f1_curve[best_idx])

TABLE_T = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
rows = []
for t in TABLE_T:
    preds = (p_ens >= t).astype(int)
    rows.append({
        "Threshold": t,
        "Precision": precision_score(y_te, preds, zero_division=0),
        "Recall":    recall_score(y_te, preds, zero_division=0),
        "F1":        f1_score(y_te, preds, zero_division=0),
        "Pred+":     int(preds.sum()),
    })
table    = pd.DataFrame(rows)
best_tbl = table.loc[table["F1"].idxmax()]

print("  " + "─" * 68)
print(f"  {'Thresh':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Pred+':>7}  Note")
print("  " + "─" * 68)
for _, row in table.iterrows():
    is_best    = row["Threshold"] == best_tbl["Threshold"]
    is_default = row["Threshold"] == 0.50
    note = " ◄ best F1" if is_best and not is_default else (" ◄ default" if is_default else "")
    print(f"  {row['Threshold']:>6.2f}  {row['Precision']:>9.4f}  "
          f"{row['Recall']:>7.4f}  {row['F1']:>7.4f}  "
          f"{int(row['Pred+']):>7}  {note}")
print("  " + "─" * 68)
print(f"\n  ★ Best F1 (table)      : {best_tbl['F1']:.4f}  at t={best_tbl['Threshold']}")
print(f"  ★ Best F1 (full curve) : {best_f1:.4f}  at t≈{best_t:.3f}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"
C_XGB    = "#d2a8ff"   # purple  — v9
C_LSTM   = "#ff7b72"   # coral   — LSTM
C_ENS    = "#3fb950"   # green   — ensemble
C_BASE   = "#388bfd"   # blue    — baseline
MARKER_COLS = ["#d29922", "#79c0ff", "#f85149",
               "#3fb950", "#bc8cff", "#ffa657", "#ff7b72"]

# Individual model PR curves (approximate — recompute on same test set)
p_xgb_ap = average_precision_score(y_te, p_xgb)
p_lstm_ap = average_precision_score(y_te, p_lstm)

px_prec, px_rec, _ = precision_recall_curve(y_te, p_xgb)
pl_prec, pl_rec, _ = precision_recall_curve(y_te, p_lstm)

fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

# Background individual models
ax.plot(px_rec, px_prec, color=C_XGB, lw=1.5, alpha=0.55, ls="--",
        label=f"XGBoost v9          AP = {p_xgb_ap:.3f}")
ax.plot(pl_rec, pl_prec, color=C_LSTM, lw=1.5, alpha=0.55, ls="--",
        label=f"LSTM (5-lap seq)    AP = {p_lstm_ap:.3f}")

# Ensemble (bold)
ax.plot(rec_pts, prec_pts, color=C_ENS, lw=2.8, zorder=4,
        label=f"Ensemble (0.5+0.5)  AP = {ap:.3f}")
ax.fill_between(rec_pts, prec_pts, alpha=0.10, color=C_ENS, zorder=1)

ax.axhline(baseline, color=C_BASE, lw=1.3, ls="--",
           label=f"Baseline ({baseline*100:.1f}%)", zorder=2)

# Best F1 marker
ax.scatter(rec_pts[best_idx], prec_pts[best_idx], color="#f0883e", s=150,
           zorder=7, edgecolors="white", lw=1.2,
           label=f"Best F1={best_f1:.3f}  (t≈{best_t:.3f})")
ax.plot([rec_pts[best_idx]]*2, [0, prec_pts[best_idx]], color="#f0883e",
        lw=0.8, ls=":", alpha=0.6)
ax.plot([0, rec_pts[best_idx]], [prec_pts[best_idx]]*2, color="#f0883e",
        lw=0.8, ls=":", alpha=0.6)

# Threshold markers
for t, col in zip(TABLE_T, MARKER_COLS):
    idx = min(np.searchsorted(thresh_pts, t), len(prec_pts) - 2)
    ax.scatter(rec_pts[idx], prec_pts[idx], color=col, s=85, zorder=6,
               edgecolors="white", lw=0.8)
    ax.annotate(f"t={t}", xy=(rec_pts[idx], prec_pts[idx]),
                xytext=(rec_pts[idx]+0.012, prec_pts[idx]+0.018),
                fontsize=8.5, color=col, fontweight="bold")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Recall", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_ylabel("Precision", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_title(
    "Precision–Recall Curve  ·  Ensemble (XGBoost v9 + LSTM)  ·  2023",
    fontsize=13, color=TEXT_COL, pad=14, fontweight="bold",
)
ax.tick_params(colors=MUTED, labelsize=9.5)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, lw=0.6, ls="--", alpha=0.7)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=10.5, facecolor=DARK_BG, edgecolor=GRID_COL,
          labelcolor=TEXT_COL, loc="upper right")

ax.text(0.02, 0.16,
        f"ROC-AUC: v9={V9['roc']:.4f}  LSTM={LSTM_B['roc']:.4f}  "
        f"Ensemble={roc_auc:.4f}  |  Lift: {ap/baseline:.1f}×",
        transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")
fig.text(0.5, 0.01,
         "Orange ★ = threshold maximising F1 for pit class (continuous PR curve).",
         ha="center", fontsize=9, color=MUTED, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("precision_recall_curve_ensemble.png", dpi=160,
            bbox_inches="tight", facecolor=DARK_BG)
print("  Plot saved → precision_recall_curve_ensemble.png  ✓")
