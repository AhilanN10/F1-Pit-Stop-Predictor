"""
Precision-Recall Curve v3 — F1 2023
=====================================
Loads pit_model_v3.json and the 10-feature test set, produces a styled
PR curve, threshold table, and identifies the F1-maximising threshold.
Compares AP against v2 (0.1335).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

V2_AP = 0.1335

# ── Rebuild test set (identical seed/steps to train_model_v3.py) ──────────────
df = pd.read_csv("f1_pit_data_2023.csv")
df = df.dropna(subset=["rolling_avg_time", "degradation_rate", "position"])
df["gap_ahead"]  = df["gap_ahead"].fillna(999.0)
df["gap_behind"] = df["gap_behind"].fillna(999.0)

FEATURE_COLS = [
    "compound", "tire_age", "rolling_avg_time", "degradation_rate",
    "gap_ahead", "gap_behind", "position", "race_completion",
    "undercut_threat", "circuit_id",
]
LABEL_COL = "pitted_next_lap"

X = df[FEATURE_COLS]
y = df[LABEL_COL]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Load model & score ────────────────────────────────────────────────────────
model = xgb.XGBClassifier()
model.load_model("pit_model_v3.json")
y_proba = model.predict_proba(X_test)[:, 1]

# ── PR curve ──────────────────────────────────────────────────────────────────
precision_pts, recall_pts, thresholds_pts = precision_recall_curve(y_test, y_proba)
ap       = average_precision_score(y_test, y_proba)
baseline = y_test.mean()

# ── Threshold table ───────────────────────────────────────────────────────────
TABLE_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
rows = []
for t in TABLE_THRESHOLDS:
    preds = (y_proba >= t).astype(int)
    rows.append({
        "Threshold":           t,
        "Precision":           precision_score(y_test, preds, zero_division=0),
        "Recall":              recall_score(y_test, preds, zero_division=0),
        "F1":                  f1_score(y_test, preds, zero_division=0),
        "Predicted Positives": int(preds.sum()),
    })
table = pd.DataFrame(rows)

# Find the threshold in the table that maximises F1
best_row   = table.loc[table["F1"].idxmax()]
best_t     = best_row["Threshold"]
best_f1    = best_row["F1"]
best_prec  = best_row["Precision"]
best_rec   = best_row["Recall"]
best_npos  = int(best_row["Predicted Positives"])

# Also scan the full curve for the continuous F1 maximum
f1_curve   = 2 * precision_pts * recall_pts / np.maximum(precision_pts + recall_pts, 1e-9)
best_curve_idx = np.argmax(f1_curve[:-1])   # exclude sentinel point at end
best_curve_t   = thresholds_pts[best_curve_idx]
best_curve_f1  = f1_curve[best_curve_idx]

# ── Print ─────────────────────────────────────────────────────────────────────
ap_delta = ap - V2_AP
ap_sign  = "+" if ap_delta >= 0 else ""

print()
print("  ┌─────────────────────────────────────────────────┐")
print("  │         PRECISION-RECALL ANALYSIS  v3           │")
print("  └─────────────────────────────────────────────────┘")
print(f"  Average Precision (v3)  : {ap:.4f}")
print(f"  Average Precision (v2)  : {V2_AP:.4f}")
print(f"  Δ AP  v3 vs v2          : {ap_sign}{ap_delta:.4f}")
print(f"  Baseline (random)       : {baseline:.4f}")
print(f"  Lift over baseline      : {ap/baseline:.1f}×")
print()
print("  " + "─" * 66)
print(f"  {'Threshold':>9}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Pred Pos':>10}  {'Note'}")
print("  " + "─" * 66)
for _, row in table.iterrows():
    is_best = (row["Threshold"] == best_t)
    marker  = " ◄ best F1" if is_best else (" ◄ default" if row["Threshold"] == 0.50 else "")
    print(f"  {row['Threshold']:>9.2f}  {row['Precision']:>9.4f}  "
          f"{row['Recall']:>7.4f}  {row['F1']:>7.4f}  "
          f"{int(row['Predicted Positives']):>10}  {marker}")
print("  " + "─" * 66)
print()
print(f"  ★ Best F1 in table    : {best_f1:.4f}  at t={best_t}")
print(f"    Precision={best_prec:.4f}  Recall={best_rec:.4f}  Pred+={best_npos}")
print()
print(f"  ★ Best F1 on full curve : {best_curve_f1:.4f}  at t≈{best_curve_t:.3f}")
print()

# ── Plot ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#bc8cff"     # purple for v3
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"
BASELINE_COL = "#388bfd"
MARKER_COLS  = ["#3fb950", "#d29922", "#f85149",
                "#79c0ff", "#ffa657", "#ff7b72", "#bc8cff"]

fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

# PR curve
ax.plot(recall_pts, precision_pts, color=ACCENT, linewidth=2.5,
        label=f"XGBoost v3 + circuit_id  (AP = {ap:.3f})", zorder=3)
ax.fill_between(recall_pts, precision_pts, alpha=0.12, color=ACCENT, zorder=1)

# Baseline
ax.axhline(baseline, color=BASELINE_COL, linewidth=1.4, linestyle="--",
           label=f"Baseline  ({baseline*100:.1f}%)", zorder=2)

# Best-F1 crosshair on the continuous curve
best_prec_c = precision_pts[best_curve_idx]
best_rec_c  = recall_pts[best_curve_idx]
ax.scatter(best_rec_c, best_prec_c, color="#f0883e", s=130, zorder=7,
           edgecolors="white", linewidths=1.2,
           label=f"Best F1={best_curve_f1:.3f}  (t≈{best_curve_t:.2f})")
ax.plot([best_rec_c, best_rec_c], [0, best_prec_c],
        color="#f0883e", linewidth=0.8, linestyle=":", alpha=0.6, zorder=6)
ax.plot([0, best_rec_c], [best_prec_c, best_prec_c],
        color="#f0883e", linewidth=0.8, linestyle=":", alpha=0.6, zorder=6)

# Threshold markers
for t, col in zip(TABLE_THRESHOLDS, MARKER_COLS):
    idx = min(np.searchsorted(thresholds_pts, t, side="left"),
              len(precision_pts) - 2)
    prec_t = precision_pts[idx]
    rec_t  = recall_pts[idx]
    ax.scatter(rec_t, prec_t, color=col, s=80, zorder=5,
               edgecolors="white", linewidths=0.8)
    yoff = 0.018 if prec_t < 0.80 else -0.035
    ax.annotate(f"t={t}", xy=(rec_t, prec_t),
                xytext=(rec_t + 0.012, prec_t + yoff),
                fontsize=8.5, color=col, fontweight="bold", zorder=6)

# Styling
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Recall", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_ylabel("Precision", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_title(
    "Precision–Recall Curve  ·  XGBoost v3 (+ circuit_id)  ·  2023 Season",
    fontsize=13, color=TEXT_COL, pad=14, fontweight="bold",
)
ax.tick_params(colors=MUTED, which="both", labelsize=9.5)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.7)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=10, facecolor=DARK_BG, edgecolor=GRID_COL,
          labelcolor=TEXT_COL, loc="upper right")

# AP comparison annotation
ax.text(0.02, 0.25,
        f"AP  v1=0.1390  v2=0.1335  v3={ap:.4f}",
        transform=ax.transAxes, fontsize=9, color=MUTED,
        style="italic", zorder=8)

fig.text(0.5, 0.01,
         "Orange ★ = threshold maximising F1 for pit class (continuous curve).",
         ha="center", fontsize=9, color=MUTED, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("precision_recall_curve_v3.png", dpi=160,
            bbox_inches="tight", facecolor=DARK_BG)
print("  Plot saved → precision_recall_curve_v3.png  ✓")
