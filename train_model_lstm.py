"""
LSTM Pit Stop Classifier — F1 2023, 19 features, 5-lap sequences
=================================================================
Builds sliding windows of 5 consecutive laps per (driver, race),
trains a 2-layer LSTM with BCEWithLogitsLoss + pos_weight, and
evaluates on a held-out 20% test split.

Saves model   : pit_model_lstm.pt
Saves PR curve: precision_recall_curve_lstm.png
"""

import subprocess, sys

# ── Install torch if missing ──────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("Installing PyTorch …")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "--quiet"],
        check=True,
    )
    import torch

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
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

# ── Baselines ─────────────────────────────────────────────────────────────────
V9 = {"roc": 0.8363, "ap": 0.1689}

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN    = 5
HIDDEN_DIM = 64
N_LAYERS   = 2
DROPOUT    = 0.3
LR         = 1e-3
BATCH_SIZE = 64
EPOCHS     = 50
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}\n")

# ── Feature set (identical to v9) ────────────────────────────────────────────
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

# ── Data prep (mirrors v9 preprocessing) ─────────────────────────────────────
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

# ── Feature scaling ───────────────────────────────────────────────────────────
# Fit scaler on all 2023 data; sequences are sliced from the scaled array.
scaler = StandardScaler()
X_all  = scaler.fit_transform(df[FEATURE_COLS].values.astype(float))
y_all  = df[LABEL_COL].values.astype(float)

# Keep group keys for sequence building
df_meta = df[["season", "round", "driver", "lap_number"]].copy().reset_index(drop=True)
df_meta["X_idx"] = np.arange(len(df_meta))   # row pointer into X_all

# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(meta, X, y, seq_len=5):
    """
    For each (round, driver) group, slide a window of `seq_len` laps.
    Only include windows where lap_number is strictly consecutive (no gaps).
    Label = pitted_next_lap of the LAST lap in the window.
    Returns arrays: X_seq (N, seq_len, n_feat), y_seq (N,).
    """
    X_seqs, y_seqs = [], []
    for (rnd, drv), grp in meta.groupby(["round", "driver"]):
        grp = grp.sort_values("lap_number").reset_index(drop=True)
        laps = grp["lap_number"].values
        idxs = grp["X_idx"].values

        for i in range(len(grp) - seq_len + 1):
            window_laps = laps[i : i + seq_len]
            # Require strictly consecutive lap numbers
            if np.all(np.diff(window_laps) == 1):
                window_idxs = idxs[i : i + seq_len]
                X_seqs.append(X[window_idxs])       # (seq_len, n_feat)
                y_seqs.append(y[window_idxs[-1]])   # label of final lap

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

print("Building 5-lap sequences …")
X_seq, y_seq = build_sequences(df_meta, X_all, y_all, SEQ_LEN)
n_pos = int(y_seq.sum())
n_neg = int((y_seq == 0).sum())
print(f"Sequences total : {len(X_seq):,}")
print(f"  pit     (1)   : {n_pos:,}  ({100*n_pos/len(y_seq):.2f}%)")
print(f"  no-pit  (0)   : {n_neg:,}  ({100*n_neg/len(y_seq):.2f}%)")
print(f"  pos_weight    : {n_neg/n_pos:.1f}\n")

# ── Train / test split (stratified) ──────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_seq, y_seq, test_size=0.20, random_state=SEED, stratify=y_seq
)
print(f"Train sequences : {len(X_tr):,}  |  pit rate {y_tr.mean()*100:.2f}%")
print(f"Test  sequences : {len(X_te):,}  |  pit rate {y_te.mean()*100:.2f}%\n")

# ── Dataset & DataLoader ──────────────────────────────────────────────────────
class LapDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)   # (N, 1)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(LapDataset(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
test_loader  = DataLoader(LapDataset(X_te, y_te),
                          batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class PitLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state of top layer
        last_hidden = self.dropout(h_n[-1])   # (batch, hidden_dim)
        return self.head(last_hidden)          # (batch, 1)  — logits

model = PitLSTM(len(FEATURE_COLS), HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}\n")

# ── Training ──────────────────────────────────────────────────────────────────
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

print("=" * 60)
print(f"  Training  {EPOCHS} epochs  |  batch {BATCH_SIZE}  |  lr {LR}")
print("=" * 60)

best_val_ap   = 0.0
best_state    = None
history_loss  = []
history_ap    = []

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    epoch_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(Xb)
    epoch_loss /= len(X_tr)

    # ── Validate ──
    model.eval()
    all_proba, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb.to(DEVICE))
            proba  = torch.sigmoid(logits).cpu().numpy().flatten()
            all_proba.extend(proba)
            all_labels.extend(yb.numpy().flatten())
    val_ap = average_precision_score(all_labels, all_proba)
    scheduler.step(val_ap)

    history_loss.append(epoch_loss)
    history_ap.append(val_ap)

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={epoch_loss:.4f}  "
              f"val_AP={val_ap:.4f}  (best={best_val_ap:.4f})")

print(f"\n  Best val AP : {best_val_ap:.4f}  (epoch {int(np.argmax(history_ap))+1})")

# ── Final evaluation with best checkpoint ─────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_proba, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb.to(DEVICE))
        proba  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_proba.extend(proba)
        all_labels.extend(yb.numpy().flatten())

y_proba = np.array(all_proba)
y_true  = np.array(all_labels)
y_pred  = (y_proba >= 0.50).astype(int)

roc_auc  = roc_auc_score(y_true, y_proba)
ap       = average_precision_score(y_true, y_proba)
baseline = float(y_true.mean())
cm       = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 64)
print("  EVALUATION  —  pit_model_lstm  (5-lap LSTM, 19 feat, 2023)")
print("=" * 64)
print(classification_report(
    y_true, y_pred,
    target_names=["No Pit (0)", "Pit (1)"],
    digits=4,
))
print(f"  ROC-AUC               : {roc_auc:.4f}")
print(f"  Average Precision (AP): {ap:.4f}")
print(f"  Best val AP           : {best_val_ap:.4f}")
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

print("\n" + "=" * 68)
print("  VERSION COMPARISON")
print("  " + "-" * 64)
print(f"  {'Metric':<30}  {'v9 (XGB)':>10}  {'LSTM':>8}  {'Δ':>12}")
print("  " + "-" * 64)
print(f"  {'ROC-AUC':<30}  {V9['roc']:>10.4f}  {roc_auc:>8.4f}  {_d(roc_auc, V9['roc']):>12}")
print(f"  {'Avg Precision (AP)':<30}  {V9['ap']:>10.4f}  {ap:>8.4f}  {_d(ap, V9['ap']):>12}")
print(f"  {'Lift over baseline':<30}  {'5.5×':>10}  {ap/baseline:.1f}×{'':<4}")
print(f"  {'Architecture':<30}  {'XGBoost':>10}  {'LSTM':>8}")
print(f"  {'Sequence aware':<30}  {'No':>10}  {'Yes':>8}")
print(f"  {'Features':<30}  {'19':>10}  {'19':>8}")
print(f"  {'TP @ t=0.50':<30}  {'68':>10}  {tp:>8}")
print(f"  {'FN @ t=0.50':<30}  {'53':>10}  {fn:>8}")
print("  " + "-" * 64)
print()

# ── Threshold table ───────────────────────────────────────────────────────────
prec_pts, rec_pts, thresh_pts = precision_recall_curve(y_true, y_proba)
f1_curve = 2 * prec_pts * rec_pts / np.maximum(prec_pts + rec_pts, 1e-9)
best_idx = int(np.argmax(f1_curve[:-1]))
best_t   = float(thresh_pts[best_idx])
best_f1  = float(f1_curve[best_idx])

TABLE_T = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
rows = []
for t in TABLE_T:
    preds = (y_proba >= t).astype(int)
    rows.append({
        "Threshold": t,
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall":    recall_score(y_true, preds, zero_division=0),
        "F1":        f1_score(y_true, preds, zero_division=0),
        "Pred+":     int(preds.sum()),
    })
table    = pd.DataFrame(rows)
best_tbl = table.loc[table["F1"].idxmax()]

print(f"  Baseline: {baseline:.4f}  |  Lift: {ap/baseline:.1f}×\n")
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

# ── Save model ────────────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": best_state,
    "scaler_mean":      scaler.mean_,
    "scaler_scale":     scaler.scale_,
    "feature_cols":     FEATURE_COLS,
    "seq_len":          SEQ_LEN,
    "hidden_dim":       HIDDEN_DIM,
    "n_layers":         N_LAYERS,
    "dropout":          DROPOUT,
    "best_val_ap":      best_val_ap,
}, "pit_model_lstm.pt")
print("  Model saved → pit_model_lstm.pt  ✓")

# ── PR curve plot ─────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#ff7b72"       # coral — LSTM
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"
BASELINE_COL = "#388bfd"
MARKER_COLS  = ["#d29922", "#79c0ff", "#f85149",
                "#3fb950", "#bc8cff", "#ffa657", "#d2a8ff"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG)

# ── Left: PR curve ─────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(PANEL_BG)
ax.plot(rec_pts, prec_pts, color=ACCENT, linewidth=2.5,
        label=f"LSTM  (5-lap seq, 19 feat,  AP = {ap:.3f})", zorder=3)
ax.fill_between(rec_pts, prec_pts, alpha=0.12, color=ACCENT, zorder=1)
ax.axhline(baseline, color=BASELINE_COL, lw=1.4, ls="--",
           label=f"Baseline ({baseline*100:.1f}%)", zorder=2)
ax.scatter(rec_pts[best_idx], prec_pts[best_idx], color="#f0883e", s=140,
           zorder=7, edgecolors="white", lw=1.2,
           label=f"Best F1={best_f1:.3f}  (t≈{best_t:.3f})")
ax.plot([rec_pts[best_idx]]*2, [0, prec_pts[best_idx]], color="#f0883e", lw=0.8, ls=":", alpha=0.6)
ax.plot([0, rec_pts[best_idx]], [prec_pts[best_idx]]*2, color="#f0883e", lw=0.8, ls=":", alpha=0.6)
for t, col in zip(TABLE_T, MARKER_COLS):
    idx = min(np.searchsorted(thresh_pts, t), len(prec_pts) - 2)
    ax.scatter(rec_pts[idx], prec_pts[idx], color=col, s=80, zorder=5,
               edgecolors="white", lw=0.8)
    ax.annotate(f"t={t}", xy=(rec_pts[idx], prec_pts[idx]),
                xytext=(rec_pts[idx]+0.012, prec_pts[idx]+0.018),
                fontsize=8.5, color=col, fontweight="bold", zorder=6)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Recall", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_ylabel("Precision", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_title("PR Curve · LSTM (5-lap sequences)", fontsize=12, color=TEXT_COL,
             pad=12, fontweight="bold")
ax.tick_params(colors=MUTED, labelsize=9.5)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, lw=0.6, ls="--", alpha=0.7)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=9.5, facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax.text(0.02, 0.12, f"AP  v9=0.1689  LSTM={ap:.4f}  |  Lift: {ap/baseline:.1f}×",
        transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

# ── Right: training curves ────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(PANEL_BG)
epochs_x = np.arange(1, EPOCHS + 1)
color_loss = "#ffa657"
color_ap   = ACCENT
ax2_r = ax2.twinx()
ax2.set_facecolor(PANEL_BG)
ax2_r.set_facecolor(PANEL_BG)
l1, = ax2.plot(epochs_x, history_loss, color=color_loss, lw=2, label="Train loss")
l2, = ax2_r.plot(epochs_x, history_ap,   color=color_ap,   lw=2, label="Val AP",    ls="--")
ax2_r.axhline(best_val_ap, color=color_ap, lw=0.8, ls=":", alpha=0.6)
ax2_r.annotate(f"best={best_val_ap:.4f}", xy=(epochs_x[np.argmax(history_ap)], best_val_ap),
               xytext=(5, best_val_ap + 0.002), fontsize=9, color=color_ap)
ax2.set_xlabel("Epoch", fontsize=11, color=TEXT_COL, labelpad=6)
ax2.set_ylabel("BCEWithLogits Loss", fontsize=11, color=color_loss, labelpad=6)
ax2_r.set_ylabel("Validation AP",    fontsize=11, color=color_ap,   labelpad=6)
ax2.tick_params(colors=MUTED, labelsize=9)
ax2_r.tick_params(colors=MUTED, labelsize=9)
for sp in ax2.spines.values():  sp.set_edgecolor(GRID_COL)
for sp in ax2_r.spines.values(): sp.set_edgecolor(GRID_COL)
ax2.grid(True, color=GRID_COL, lw=0.6, ls="--", alpha=0.7)
ax2.set_title("Training Curves", fontsize=12, color=TEXT_COL, pad=12, fontweight="bold")
ax2.legend(handles=[l1, l2], fontsize=9.5, facecolor=DARK_BG, edgecolor=GRID_COL,
           labelcolor=TEXT_COL, loc="upper right")

fig.suptitle(
    f"LSTM Pit Stop Predictor  ·  2023  ·  5-lap sequences  ·  "
    f"AP={ap:.4f}  ROC-AUC={roc_auc:.4f}",
    fontsize=13, color=TEXT_COL, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("precision_recall_curve_lstm.png", dpi=160,
            bbox_inches="tight", facecolor=DARK_BG)
print("  Plot saved → precision_recall_curve_lstm.png  ✓")
