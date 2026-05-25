"""
XGBoost Pit Stop Classifier v4 — F1 2023
==========================================
12-feature model: adds tire_age_squared and pit_loss_time to the v3 set.
Saves model as pit_model_v4.json.
Saves PR curve as precision_recall_curve_v4.png.
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
V1 = {"roc": 0.8219, "ap": 0.1390, "tp": 82,  "fn": 40,  "n": 8}
V2 = {"roc": 0.8188, "ap": 0.1335, "tp": 86,  "fn": 36,  "n": 9}
V3 = {"roc": 0.8328, "ap": 0.1283, "tp": 78,  "fn": 44,  "n": 10}

# ── Data prep ─────────────────────────────────────────────────────────────────
df = pd.read_csv("f1_pit_data_2023.csv")
print(f"Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}\n")

df = df.dropna(subset=["rolling_avg_time", "degradation_rate", "position"])
df["gap_ahead"]  = df["gap_ahead"].fillna(999.0)
df["gap_behind"] = df["gap_behind"].fillna(999.0)

FEATURE_COLS = [
    "compound",
    "tire_age",
    "tire_age_squared",       # ← new (quadratic degradation)
    "rolling_avg_time",
    "degradation_rate",
    "gap_ahead",
    "gap_behind",
    "position",
    "race_completion",
    "undercut_threat",
    "circuit_id",
    "pit_loss_time",          # ← new (circuit-level pit penalty)
]
LABEL_COL = "pitted_next_lap"

X = df[FEATURE_COLS]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train : {X_train.shape}  |  pit rate {y_train.mean()*100:.2f}%")
print(f"Test  : {X_test.shape}  |  pit rate {y_test.mean()*100:.2f}%\n")

# ── Train ─────────────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=32,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
ap      = average_precision_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 62)
print("  EVALUATION  —  pit_model_v4  (12 features)")
print("=" * 62)
print(classification_report(
    y_test, y_pred,
    target_names=["No Pit (0)", "Pit (1)"],
    digits=4,
))
print(f"  ROC-AUC               : {roc_auc:.4f}")
print(f"  Average Precision (AP): {ap:.4f}")
print()
print("  Confusion Matrix")
print("                   Predicted 0   Predicted 1")
print(f"  Actual 0  (no pit)   {cm[0,0]:>6}        {cm[0,1]:>6}")
print(f"  Actual 1  (pit)      {cm[1,0]:>6}        {cm[1,1]:>6}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  Best iteration : {model.best_iteration}")
print("=" * 62)

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(
    model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\n  FEATURE IMPORTANCES  (weight)")
print("  " + "-" * 50)
for rank, (feat, score) in enumerate(importances.items(), 1):
    new = " ← new" if feat in ("tire_age_squared", "pit_loss_time") else ""
    bar = "█" * int(score * 300)
    print(f"  {rank:>2}. {feat:<22}  {score:.4f}  {bar}{new}")

# ── Version comparison ────────────────────────────────────────────────────────
def _d(new, old, pct=False):
    diff = new - old
    s = f"{diff:+.4f}" if not pct else f"{diff:+.1f}%"
    return s

print("\n" + "=" * 68)
print("  VERSION COMPARISON")
print("  " + "-" * 64)
print(f"  {'Metric':<28}  {'v1':>7}  {'v2':>7}  {'v3':>7}  {'v4':>7}  {'v4 vs v3':>10}")
print("  " + "-" * 64)
print(f"  {'ROC-AUC':<28}  {V1['roc']:.4f}  {V2['roc']:.4f}  {V3['roc']:.4f}  {roc_auc:.4f}  {_d(roc_auc,V3['roc']):>10}")
print(f"  {'Avg Precision (AP)':<28}  {V1['ap']:.4f}  {V2['ap']:.4f}  {V3['ap']:.4f}  {ap:.4f}  {_d(ap,V3['ap']):>10}")
print(f"  {'TP @ t=0.50':<28}  {V1['tp']:>7}  {V2['tp']:>7}  {V3['tp']:>7}  {tp:>7}  {tp-V3['tp']:>+10}")
print(f"  {'FN @ t=0.50':<28}  {V1['fn']:>7}  {V2['fn']:>7}  {V3['fn']:>7}  {fn:>7}  {fn-V3['fn']:>+10}")
print(f"  {'Features':<28}  {V1['n']:>7}  {V2['n']:>7}  {V3['n']:>7}  {'12':>7}  {'':>10}")
print("  " + "-" * 64)
print()

# ── Save model ────────────────────────────────────────────────────────────────
model.save_model("pit_model_v4.json")
print("  Model saved → pit_model_v4.json  ✓")

# ── PR curve & threshold table ────────────────────────────────────────────────
precision_pts, recall_pts, thresholds_pts = precision_recall_curve(y_test, y_proba)
baseline = y_test.mean()

# F1 on full curve
f1_curve       = 2 * precision_pts * recall_pts / np.maximum(precision_pts + recall_pts, 1e-9)
best_curve_idx = int(np.argmax(f1_curve[:-1]))
best_curve_t   = float(thresholds_pts[best_curve_idx])
best_curve_f1  = float(f1_curve[best_curve_idx])

TABLE_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
rows = []
for t in TABLE_THRESHOLDS:
    preds = (y_proba >= t).astype(int)
    rows.append({
        "Threshold": t,
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall":    recall_score(y_test, preds, zero_division=0),
        "F1":        f1_score(y_test, preds, zero_division=0),
        "Pred+":     int(preds.sum()),
    })
table = pd.DataFrame(rows)
best_tbl_row = table.loc[table["F1"].idxmax()]

print(f"\n  AP (v4)={ap:.4f}  vs  AP (v3)={V3['ap']:.4f}  Δ={_d(ap,V3['ap'])}")
print(f"  Baseline: {baseline:.4f}  |  Lift: {ap/baseline:.1f}×\n")
print("  " + "─" * 66)
print(f"  {'Thresh':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Pred+':>7}  Note")
print("  " + "─" * 66)
for _, row in table.iterrows():
    is_best    = row["Threshold"] == best_tbl_row["Threshold"]
    is_default = row["Threshold"] == 0.50
    note = " ◄ best F1" if is_best and not is_default else (" ◄ default" if is_default else "")
    print(f"  {row['Threshold']:>6.2f}  {row['Precision']:>9.4f}  "
          f"{row['Recall']:>7.4f}  {row['F1']:>7.4f}  "
          f"{int(row['Pred+']):>7}  {note}")
print("  " + "─" * 66)
print(f"\n  ★ Best F1 (table)        : {best_tbl_row['F1']:.4f}  at t={best_tbl_row['Threshold']}")
print(f"  ★ Best F1 (full curve)   : {best_curve_f1:.4f}  at t≈{best_curve_t:.3f}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#ff7b72"      # red/coral for v4
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"
BASELINE_COL = "#388bfd"
MARKER_COLS  = ["#3fb950", "#d29922", "#f85149",
                "#79c0ff", "#bc8cff", "#ffa657", "#ff7b72"]

fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

ax.plot(recall_pts, precision_pts, color=ACCENT, linewidth=2.5,
        label=f"XGBoost v4  (12 features,  AP = {ap:.3f})", zorder=3)
ax.fill_between(recall_pts, precision_pts, alpha=0.12, color=ACCENT, zorder=1)
ax.axhline(baseline, color=BASELINE_COL, linewidth=1.4, linestyle="--",
           label=f"Baseline  ({baseline*100:.1f}%)", zorder=2)

# Best-F1 crosshair
bpc = precision_pts[best_curve_idx]
brc = recall_pts[best_curve_idx]
ax.scatter(brc, bpc, color="#f0883e", s=140, zorder=7,
           edgecolors="white", linewidths=1.2,
           label=f"Best F1={best_curve_f1:.3f}  (t≈{best_curve_t:.2f})")
ax.plot([brc, brc], [0, bpc], color="#f0883e", lw=0.8, ls=":", alpha=0.6, zorder=6)
ax.plot([0, brc], [bpc, bpc], color="#f0883e", lw=0.8, ls=":", alpha=0.6, zorder=6)

# Threshold markers
for t, col in zip(TABLE_THRESHOLDS, MARKER_COLS):
    idx = min(np.searchsorted(thresholds_pts, t, side="left"), len(precision_pts) - 2)
    pt, rt = precision_pts[idx], recall_pts[idx]
    ax.scatter(rt, pt, color=col, s=80, zorder=5, edgecolors="white", linewidths=0.8)
    yoff = 0.018 if pt < 0.80 else -0.035
    ax.annotate(f"t={t}", xy=(rt, pt), xytext=(rt + 0.012, pt + yoff),
                fontsize=8.5, color=col, fontweight="bold", zorder=6)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Recall", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_ylabel("Precision", fontsize=12, color=TEXT_COL, labelpad=8)
ax.set_title(
    "Precision–Recall Curve  ·  XGBoost v4 (+ tire_age² + pit_loss_time)  ·  2023 Season",
    fontsize=12.5, color=TEXT_COL, pad=14, fontweight="bold",
)
ax.tick_params(colors=MUTED, labelsize=9.5)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.7)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(fontsize=10, facecolor=DARK_BG, edgecolor=GRID_COL,
          labelcolor=TEXT_COL, loc="upper right")

ax.text(0.02, 0.22,
        f"AP  v1=0.1390  v2=0.1335  v3=0.1283  v4={ap:.4f}",
        transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")
fig.text(0.5, 0.01,
         "Orange ★ = threshold maximising F1 for pit class (continuous curve).",
         ha="center", fontsize=9, color=MUTED, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("precision_recall_curve_v4.png", dpi=160,
            bbox_inches="tight", facecolor=DARK_BG)
print("  Plot saved → precision_recall_curve_v4.png  ✓")
