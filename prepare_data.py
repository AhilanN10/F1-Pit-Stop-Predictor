"""
ML Data Preparation — F1 Pit Stop Prediction
=============================================
Loads f1_pit_data_2023.csv, applies cleaning steps, splits into
train/test sets stratified on the pit label, and reports shapes
and class balance for both splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("f1_pit_data_2023.csv")
print(f"Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Cleaning ───────────────────────────────────────────────────────────────────

# 1. Drop rows where rolling features are null (first 2 laps of each stint)
before = len(df)
df = df.dropna(subset=["rolling_avg_time", "degradation_rate"])
print(f"After dropping null rolling_avg_time / degradation_rate : {len(df):,} rows  (dropped {before - len(df):,})")

# 2. Drop rows where position is null
before = len(df)
df = df.dropna(subset=["position"])
print(f"After dropping null position                            : {len(df):,} rows  (dropped {before - len(df):,})")

# 3. Fill gap nulls with 999.0 (sentinel for leader / last-place)
df["gap_ahead"]  = df["gap_ahead"].fillna(999.0)
df["gap_behind"] = df["gap_behind"].fillna(999.0)
print(f"gap_ahead  nulls remaining : {df['gap_ahead'].isnull().sum()}")
print(f"gap_behind nulls remaining : {df['gap_behind'].isnull().sum()}")

# 4. Drop identifier columns
id_cols = ["year", "round", "event_name", "driver", "lap_number"]
df = df.drop(columns=id_cols)
print(f"\nDropped identifier columns : {id_cols}")
print(f"Remaining columns          : {df.columns.tolist()}")

# ── Features / Label ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "compound",
    "tire_age",
    "rolling_avg_time",
    "degradation_rate",
    "gap_ahead",
    "gap_behind",
    "position",
    "race_completion",
]
LABEL_COL = "pitted_next_lap"

X = df[FEATURE_COLS]
y = df[LABEL_COL]

print(f"\nFeature matrix X : {X.shape}")
print(f"Label vector   y : {y.shape}")
print(f"Overall pit rate : {y.mean():.4f}  ({y.sum():,} pit laps / {len(y):,} total)")

# ── Train / Test Split (stratified 80/20) ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ── Report ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("  SPLIT SUMMARY")
print("=" * 52)
print(f"  X_train : {str(X_train.shape):<18}  y_train : {str(y_train.shape)}")
print(f"  X_test  : {str(X_test.shape):<18}  y_test  : {str(y_test.shape)}")
print("=" * 52)
print(f"  Pit rate — overall : {y.mean()*100:.2f}%")
print(f"  Pit rate — train   : {y_train.mean()*100:.2f}%  ({y_train.sum():,} / {len(y_train):,})")
print(f"  Pit rate — test    : {y_test.mean()*100:.2f}%  ({y_test.sum():,} / {len(y_test):,})")
print("=" * 52)
print("\nStratification preserved ✓" if abs(y_train.mean() - y_test.mean()) < 0.001 else "\nWARNING: pit rates differ by more than 0.1%")
