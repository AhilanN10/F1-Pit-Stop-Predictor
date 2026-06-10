"""
XGBoost Pit Stop Classifier v8 — Optuna HPO (2023 only, 14 features)
=====================================================================
Runs 100 Optuna trials optimising Average Precision via 5-fold
stratified cross-validation, then retrains on full train split with
the best params and evaluates on held-out test set.

Saves model as pit_model_v8.json.
Saves PR curve as precision_recall_curve_v8.png.
"""

import subprocess, sys
# Install optuna if not present
subprocess.run(
    [sys.executable, "-m", "pip", "install", "optuna", "--quiet"],
    check=True,
)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Baselines ─────────────────────────────────────────────────────────────────
V4 = {"roc": 0.8267, "ap": 0.1446}
V7 = {"roc": 0.8130, "ap": 0.1470}

# ── Data prep ─────────────────────────────────────────────────────────────────
raw = pd.read_csv("f1_pit_data_2021_2023.csv")
df  = raw[raw["season"] == 2023].copy()
print(f"2023 rows : {len(df):,}  (of {len(raw):,} total)")

df = df.dropna(subset=["rolling_avg_time", "degradation_rate", "position"])
df["gap_ahead"]  = df["gap_ahead"].fillna(999.0)
df["gap_behind"] = df["gap_behind"].fillna(999.0)

FEATURE_COLS = [
    "compound", "tire_age", "tire_age_squared",
    "rolling_avg_time", "degradation_rate",
    "gap_ahead", "gap_behind", "position",
    "race_completion", "undercut_threat",
    "circuit_id", "pit_loss_time",
    "track_temp", "rainfall",
]
LABEL_COL = "pitted_next_lap"

X = df[FEATURE_COLS].values
y = df[LABEL_COL].values

n_neg = int((y == 0).sum())
n_pos = int((y == 1).sum())
SPW   = round(n_neg / n_pos)          # fixed for all trials
print(f"Class ratio: {n_neg:,} / {n_pos:,}  →  scale_pos_weight = {SPW}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train : {X_train.shape}  |  pit rate {y_train.mean()*100:.2f}%")
print(f"Test  : {X_test.shape}  |  pit rate {y_test.mean()*100:.2f}%\n")

# ── Optuna objective ──────────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        # fixed
        "scale_pos_weight": SPW,
        "eval_metric":      "aucpr",
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }

    fold_aps = []
    for fold_i, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        clf = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        clf.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )
        proba = clf.predict_proba(Xval)[:, 1]
        fold_aps.append(average_precision_score(yval, proba))

        # Pruning: report intermediate value after each fold
        trial.report(np.mean(fold_aps), step=fold_i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(fold_aps))

# ── Run optimisation ──────────────────────────────────────────────────────────
print("=" * 64)
print(f"  Running Optuna HPO — 100 trials, 5-fold CV, metric = AP")
print("=" * 64)

sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
study   = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)
study.optimize(objective, n_trials=100, show_progress_bar=False)

best  = study.best_trial
best_params = best.params
best_cv_ap  = best.value

print(f"\n  ✓ Optimisation complete  ({len(study.trials)} trials run)")
print(f"  Best CV AP : {best_cv_ap:.4f}")
print(f"  Pruned     : {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)} trials")
print()
print("  BEST HYPERPARAMETERS")
print("  " + "-" * 48)
for k, v in best_params.items():
    print(f"    {k:<22} : {v}")
print("  " + "-" * 48)

# Top-5 trials by AP
trials_df = study.trials_dataframe().dropna(subset=["value"])
top5 = trials_df.nlargest(5, "value")[
    ["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]
]
print(f"\n  Top-5 trials by CV AP:")
print("  " + "-" * 48)
for _, row in top5.iterrows():
    print(f"    Trial {int(row['number']):>3}  AP={row['value']:.4f}")
print()

# ── Retrain on full training set with best params ─────────────────────────────
print("Retraining on full training split with best params ...")
final_params = {
    **best_params,
    "scale_pos_weight": SPW,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
}
model = xgb.XGBClassifier(**final_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

roc_auc  = roc_auc_score(y_test, y_proba)
ap       = average_precision_score(y_test, y_proba)
baseline = float(y_test.mean())
cm       = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 64)
print("  EVALUATION  —  pit_model_v8  (Optuna HPO, 14 feat, 2023)")
print("=" * 64)
print(classification_report(
    y_test, y_pred,
    target_names=["No Pit (0)", "Pit (1)"],
    digits=4,
))
print(f"  ROC-AUC               : {roc_auc:.4f}")
print(f"  Average Precision (AP): {ap:.4f}")
print(f"  Best CV AP (Optuna)   : {best_cv_ap:.4f}")
print(f"  Baseline (random)     : {baseline:.4f}")
print(f"  Lift over baseline    : {ap/baseline:.1f}×")
print()
print("  Confusion Matrix")
print("                   Predicted 0   Predicted 1")
print(f"  Actual 0  (no pit)   {cm[0,0]:>6}        {cm[0,1]:>6}")
print(f"  Actual 1  (pit)      {cm[1,0]:>6}        {cm[1,1]:>6}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  Best iteration : {model.best_iteration}")
print("=" * 64)

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(
    model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\n  FEATURE IMPORTANCES  (weight)")
print("  " + "-" * 56)
for rank, (feat, score) in enumerate(importances.items(), 1):
    bar = "█" * int(score * 300)
    print(f"  {rank:>2}. {feat:<22}  {score:.4f}  {bar}")

# ── Version comparison ────────────────────────────────────────────────────────
def _d(new, old):
    diff = new - old
    m = "↑" if diff > 0.0001 else "↓" if diff < -0.0001 else "≈"
    return f"{diff:+.4f} {m}"

print("\n" + "=" * 72)
print("  VERSION COMPARISON  (2023-only versions + weather, all 14 feat)")
print("  " + "-" * 68)
print(f"  {'Metric':<30}  {'v4':>8}  {'v7':>8}  {'v8':>8}  {'v8 vs v4':>12}")
print("  " + "-" * 68)
print(f"  {'ROC-AUC':<30}  {V4['roc']:>8.4f}  {V7['roc']:>8.4f}  {roc_auc:>8.4f}  {_d(roc_auc, V4['roc']):>12}")
print(f"  {'Avg Precision (AP)':<30}  {V4['ap']:>8.4f}  {V7['ap']:>8.4f}  {ap:>8.4f}  {_d(ap, V4['ap']):>12}")
print(f"  {'Best CV AP (Optuna)':<30}  {'N/A':>8}  {'N/A':>8}  {best_cv_ap:>8.4f}")
print(f"  {'Lift over baseline':<30}  {'4.7×':>8}  {'4.7×':>8}  {ap/baseline:.1f}×{'':<5}  {'':>12}")
print(f"  {'HPO':<30}  {'No':>8}  {'No':>8}  {'Optuna':>8}")
print(f"  {'TP @ t=0.50':<30}  {'84':>8}  {'82':>8}  {tp:>8}  {tp-84:>+12}")
print(f"  {'FN @ t=0.50':<30}  {'38':>8}  {'39':>8}  {fn:>8}  {fn-38:>+12}")
print("  " + "-" * 68)
print()

# ── Save model ────────────────────────────────────────────────────────────────
model.save_model("pit_model_v8.json")
print("  Model saved → pit_model_v8.json  ✓")

# ── PR curve & threshold table ────────────────────────────────────────────────
precision_pts, recall_pts, thresholds_pts = precision_recall_curve(y_test, y_proba)

f1_curve       = 2 * precision_pts * recall_pts / np.maximum(precision_pts + recall_pts, 1e-9)
best_curve_idx = int(np.argmax(f1_curve[:-1]))
best_curve_t   = float(thresholds_pts[best_curve_idx])
best_curve_f1  = float(f1_curve[best_curve_idx])

TABLE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
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

print(f"\n  AP (v8)={ap:.4f}  vs  AP (v4)={V4['ap']:.4f}  Δ={_d(ap, V4['ap'])}")
print(f"  AP (v8)={ap:.4f}  vs  AP (v7)={V7['ap']:.4f}  Δ={_d(ap, V7['ap'])}")
print(f"  Baseline: {baseline:.4f}  |  Lift: {ap/baseline:.1f}×\n")
print("  " + "─" * 68)
print(f"  {'Thresh':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Pred+':>7}  Note")
print("  " + "─" * 68)
for _, row in table.iterrows():
    is_best    = row["Threshold"] == best_tbl_row["Threshold"]
    is_default = row["Threshold"] == 0.50
    note = " ◄ best F1" if is_best and not is_default else (" ◄ default" if is_default else "")
    print(f"  {row['Threshold']:>6.2f}  {row['Precision']:>9.4f}  "
          f"{row['Recall']:>7.4f}  {row['F1']:>7.4f}  "
          f"{int(row['Pred+']):>7}  {note}")
print("  " + "─" * 68)
print(f"\n  ★ Best F1 (table)      : {best_tbl_row['F1']:.4f}  at t={best_tbl_row['Threshold']}")
print(f"  ★ Best F1 (full curve) : {best_curve_f1:.4f}  at t≈{best_curve_t:.3f}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#3fb950"       # green = optimised / best
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"
BASELINE_COL = "#388bfd"
MARKER_COLS  = ["#d29922", "#79c0ff", "#f85149",
                "#bc8cff", "#ffa657", "#ff7b72", "#3fb950"]

fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

ax.plot(recall_pts, precision_pts, color=ACCENT, linewidth=2.5,
        label=f"XGBoost v8  (Optuna HPO, 14 feat, 2023,  AP = {ap:.3f})", zorder=3)
ax.fill_between(recall_pts, precision_pts, alpha=0.12, color=ACCENT, zorder=1)
ax.axhline(baseline, color=BASELINE_COL, linewidth=1.4, linestyle="--",
           label=f"Baseline  ({baseline*100:.1f}%)", zorder=2)

bpc = precision_pts[best_curve_idx]
brc = recall_pts[best_curve_idx]
ax.scatter(brc, bpc, color="#f0883e", s=140, zorder=7,
           edgecolors="white", linewidths=1.2,
           label=f"Best F1={best_curve_f1:.3f}  (t≈{best_curve_t:.3f})")
ax.plot([brc, brc], [0, bpc], color="#f0883e", lw=0.8, ls=":", alpha=0.6, zorder=6)
ax.plot([0, brc], [bpc, bpc], color="#f0883e", lw=0.8, ls=":", alpha=0.6, zorder=6)

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
    f"Precision–Recall Curve  ·  XGBoost v8  (Optuna, 100 trials)  ·  2023",
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
        f"AP  v4=0.1446  v7=0.1470  v8={ap:.4f}  |  Best CV AP={best_cv_ap:.4f}  "
        f"Lift: {ap/baseline:.1f}×",
        transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")
fig.text(0.5, 0.01,
         "Orange ★ = threshold maximising F1 for pit class (continuous curve).",
         ha="center", fontsize=9, color=MUTED, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("precision_recall_curve_v8.png", dpi=160,
            bbox_inches="tight", facecolor=DARK_BG)
print("  Plot saved → precision_recall_curve_v8.png  ✓")
