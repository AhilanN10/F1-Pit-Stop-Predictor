#!/usr/bin/env python3
"""
F1 Pit Stop Predictor — powered by pit_model_v9.json
======================================================
Interactive CLI that takes a current race state as input and outputs
a pit stop recommendation using the trained XGBoost v9 model
(Optuna-tuned, 19 features: weather + opponent context + stint length).

Usage:
    python3 pit_predictor.py
"""

import sys
import numpy as np
import xgboost as xgb

# ── Constants ─────────────────────────────────────────────────────────────────

THRESHOLD = 0.65    # F1-maximising threshold from v9 PR curve

# Stable alphabetical circuit mapping (cross-season, 28 circuits)
CIRCUIT_NAMES = {
     0: "Abu Dhabi Grand Prix",
     1: "Australian Grand Prix",
     2: "Austrian Grand Prix",
     3: "Azerbaijan Grand Prix",
     4: "Bahrain Grand Prix",
     5: "Belgian Grand Prix",
     6: "British Grand Prix",
     7: "Canadian Grand Prix",
     8: "Dutch Grand Prix",
     9: "Emilia Romagna Grand Prix",
    10: "French Grand Prix",
    11: "Hungarian Grand Prix",
    12: "Italian Grand Prix",
    13: "Japanese Grand Prix",
    14: "Las Vegas Grand Prix",
    15: "Mexico City Grand Prix",
    16: "Miami Grand Prix",
    17: "Monaco Grand Prix",
    18: "Portuguese Grand Prix",
    19: "Qatar Grand Prix",
    20: "Russian Grand Prix",
    21: "Saudi Arabian Grand Prix",
    22: "Singapore Grand Prix",
    23: "Spanish Grand Prix",
    24: "Styrian Grand Prix",
    25: "São Paulo Grand Prix",
    26: "Turkish Grand Prix",
    27: "United States Grand Prix",
}

# Cross-season averaged pit loss times (seconds)
PIT_LOSS_TIME = {
     0:  3.39,
     1: 14.60,
     2:  3.58,
     3:  4.40,
     4:  3.01,
     5:  4.22,
     6: -0.37,
     7: 18.76,
     8:  5.61,
     9:  6.35,
    10: 19.78,
    11:  2.74,
    12:  5.00,
    13:  4.67,
    14:  6.39,
    15:  3.69,
    16:  5.01,
    17: 22.20,
    18:  3.93,
    19: 13.71,
    20: 12.23,
    21:  5.09,
    22:  9.03,
    23:  3.92,
    24:  4.87,
    25:  4.62,
    26:  6.68,
    27:  1.62,
}

# Expected stint length (laps) keyed by (circuit_id, compound)
# compound: 0=Soft  1=Medium  2=Hard
# Source: cross-season mean per (circuit_id, compound) from pipeline output.
# Missing combinations fall back to GLOBAL_MEAN_STINT_LENGTH.
GLOBAL_MEAN_STINT_LENGTH = 23.8

EXPECTED_STINT_LENGTH = {
    # (circuit_id, compound): mean laps
    ( 0, 0): 14.2,  ( 0, 1): 18.2,  ( 0, 2): 25.1,  # Abu Dhabi
    ( 1, 0):  4.5,  ( 1, 1):  9.9,  ( 1, 2): 37.0,  # Australian
    ( 2, 0): 14.6,  ( 2, 1): 23.9,  ( 2, 2): 28.1,  # Austrian
    ( 3, 0):  8.8,  ( 3, 1): 10.1,  ( 3, 2): 32.8,  # Azerbaijan
    ( 4, 0): 14.7,  ( 4, 1): 18.6,  ( 4, 2): 18.4,  # Bahrain
    ( 5, 0): 13.8,  ( 5, 1): 15.5,  ( 5, 2): 12.0,  # Belgian
    ( 6, 0): 19.4,  ( 6, 1): 24.3,  ( 6, 2): 24.3,  # British
    ( 7, 0):  6.0,  ( 7, 1): 18.4,  ( 7, 2): 31.5,  # Canadian
    ( 8, 0): 21.9,  ( 8, 1): 28.0,  ( 8, 2): 31.8,  # Dutch
    ( 9, 0): 19.5,  ( 9, 1): 26.4,                   # Emilia Romagna (no Hard data)
    (10, 0): 19.0,  (10, 1): 20.5,  (10, 2): 32.0,  # French
    (11, 0):  9.0,  (11, 1): 22.5,  (11, 2): 32.2,  # Hungarian
    (12, 0):  1.0,  (12, 1): 21.4,  (12, 2): 26.6,  # Italian
    (13, 0): 10.9,  (13, 1): 15.5,  (13, 2): 18.3,  # Japanese
    (14, 0): 10.0,  (14, 1): 16.0,  (14, 2): 23.3,  # Las Vegas
    (15, 0):  9.0,  (15, 1): 28.4,  (15, 2): 32.8,  # Mexico City
    (16, 0):  4.5,  (16, 1): 17.6,  (16, 2): 40.7,  # Miami
    (17, 0): 33.0,  (17, 1): 33.2,  (17, 2): 41.3,  # Monaco
    (18, 0): 21.6,  (18, 1): 35.8,  (18, 2): 32.8,  # Portuguese
    (19, 0): 17.1,  (19, 1): 18.8,  (19, 2): 20.4,  # Qatar
    (20, 0):  2.0,  (20, 1): 17.4,  (20, 2): 30.4,  # Russian
    (21, 0):  9.6,  (21, 1): 18.7,  (21, 2): 22.9,  # Saudi Arabian
    (22, 0): 16.5,  (22, 1): 19.9,  (22, 2): 32.5,  # Singapore
    (23, 0): 19.2,  (23, 1): 25.2,  (23, 2): 24.6,  # Spanish
    (24, 0): 20.8,  (24, 1): 29.9,  (24, 2): 36.9,  # Styrian
    (25, 0): 22.1,  (25, 1): 25.2,  (25, 2): 24.5,  # São Paulo
                    (26, 1):  5.0,                    # Turkish (Medium only)
    (27, 0):  8.8,  (27, 1): 16.4,  (27, 2): 21.0,  # United States
}

COMPOUND_NAMES = {-1: "N/A", 0: "SOFT", 1: "MEDIUM", 2: "HARD"}

FEATURE_COLS = [
    "compound",
    "tire_age",
    "tire_age_squared",
    "rolling_avg_time",
    "degradation_rate",
    "gap_ahead",
    "gap_behind",
    "position",
    "race_completion",
    "undercut_threat",
    "circuit_id",
    "pit_loss_time",
    "track_temp",
    "rainfall",
    "ahead_compound",
    "ahead_tire_age",
    "behind_compound",
    "behind_tire_age",
    "expected_stint_length",
]

# ── Styling helpers ───────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET

def header(text):
    width = 60
    pad = (width - len(text) - 2) // 2
    print()
    print(c("─" * width, CYAN))
    print(c(" " * pad + " " + text + " " + " " * pad, CYAN, BOLD))
    print(c("─" * width, CYAN))

def section(text):
    print()
    print(c(f"  ▸ {text}", YELLOW, BOLD))

def row(label, value, unit=""):
    label_str = c(f"    {label:<28}", DIM)
    value_str = c(f"{value}", WHITE, BOLD)
    unit_str  = c(f" {unit}", DIM) if unit else ""
    print(f"{label_str}{value_str}{unit_str}")

# ── Input helpers ─────────────────────────────────────────────────────────────

def prompt(label, type_=float, min_=None, max_=None, choices=None):
    """Prompt for a validated input value."""
    while True:
        try:
            raw = input(c(f"  {label}: ", CYAN)).strip()
            val = type_(raw)
            if choices is not None and val not in choices:
                print(c(f"    ✗ Must be one of: {choices}", RED))
                continue
            if min_ is not None and val < min_:
                print(c(f"    ✗ Must be ≥ {min_}", RED))
                continue
            if max_ is not None and val > max_:
                print(c(f"    ✗ Must be ≤ {max_}", RED))
                continue
            return val
        except (ValueError, KeyboardInterrupt):
            print(c("    ✗ Invalid input, please try again.", RED))

# ── Circuit selection ─────────────────────────────────────────────────────────

def choose_circuit():
    print()
    print(c("  Available circuits:", YELLOW, BOLD))
    for cid, name in CIRCUIT_NAMES.items():
        print(c(f"    {cid:>2}", CYAN) + c(f"  {name}", DIM))
    print()
    return int(prompt("Circuit ID (0–27)", int, min_=0, max_=27))

# ── Main prediction flow ──────────────────────────────────────────────────────

def collect_inputs():
    header("F1 PIT STOP PREDICTOR  ·  v9 MODEL")

    print(c("\n  Enter the current race state below.", DIM))
    print(c("  Type 999 for gap fields when leading / last place.\n", DIM))

    circuit_id    = choose_circuit()
    circuit_name  = CIRCUIT_NAMES[circuit_id]
    pit_loss_time = PIT_LOSS_TIME[circuit_id]

    section("Lap & Race State")
    current_lap = prompt("Current lap number", int, min_=1)
    total_laps  = prompt("Total race laps", int, min_=current_lap)
    position    = prompt("Current position (1–20)", int, min_=1, max_=20)

    section("Tire State")
    compound = prompt("Compound  (0=Soft  1=Medium  2=Hard)", int, choices=[0, 1, 2])
    tire_age = prompt("Tire age (laps on this set)", float, min_=0)

    section("Pace & Degradation")
    rolling_avg = prompt("3-lap rolling avg lap time (seconds)", float, min_=60)
    degradation = prompt("Degradation rate (rolling_avg - stint base, can be negative)", float)

    section("Gap Information")
    gap_ahead  = prompt("Gap to car ahead (seconds, 999 if leading)", float, min_=0)
    gap_behind = prompt("Gap to car behind (seconds, 999 if last)", float, min_=0)

    section("Opponent State  (car directly ahead & behind)")
    # Car ahead
    if gap_ahead >= 999.0:
        print(c("    (Leading — ahead opponent values auto-set to −1)", DIM))
        ahead_compound = -1
        ahead_tire_age = -1.0
    else:
        ahead_compound = prompt("Car ahead compound  (0=Soft  1=Medium  2=Hard)", int,
                                choices=[0, 1, 2])
        ahead_tire_age = prompt("Car ahead tire age (laps)", float, min_=0)

    # Car behind
    if gap_behind >= 999.0:
        print(c("    (Last place — behind opponent values auto-set to −1)", DIM))
        behind_compound = -1
        behind_tire_age = -1.0
    else:
        behind_compound = prompt("Car behind compound (0=Soft  1=Medium  2=Hard)", int,
                                 choices=[0, 1, 2])
        behind_tire_age = prompt("Car behind tire age (laps)", float, min_=0)

    section("Weather Conditions")
    track_temp = prompt("Track surface temperature (°C)", float, min_=-10, max_=80)
    rainfall   = prompt("Rainfall  (0 = dry  1 = wet)", int, choices=[0, 1])

    return {
        "circuit_id":     circuit_id,
        "circuit_name":   circuit_name,
        "pit_loss_time":  pit_loss_time,
        "current_lap":    current_lap,
        "total_laps":     total_laps,
        "position":       position,
        "compound":       compound,
        "tire_age":       tire_age,
        "rolling_avg":    rolling_avg,
        "degradation":    degradation,
        "gap_ahead":      gap_ahead,
        "gap_behind":     gap_behind,
        "ahead_compound": ahead_compound,
        "ahead_tire_age": ahead_tire_age,
        "behind_compound": behind_compound,
        "behind_tire_age": behind_tire_age,
        "track_temp":     track_temp,
        "rainfall":       rainfall,
    }


def build_feature_vector(inp):
    """Compute derived features and return ordered feature array."""
    race_completion  = inp["current_lap"] / inp["total_laps"]
    tire_age_squared = inp["tire_age"] ** 2

    # Undercut threat (uses car-behind tire age; 0 if last or sentinel)
    bta = inp["behind_tire_age"]
    if inp["gap_behind"] >= 999.0 or bta < 0:
        undercut_threat = 0.0
    else:
        denom = inp["tire_age"] - bta + 1
        undercut_threat = 0.0 if denom <= 0 else inp["gap_behind"] / denom

    # Expected stint length from hardcoded lookup table
    expected_sl = EXPECTED_STINT_LENGTH.get(
        (inp["circuit_id"], inp["compound"]),
        GLOBAL_MEAN_STINT_LENGTH,
    )

    features = {
        "compound":              inp["compound"],
        "tire_age":              inp["tire_age"],
        "tire_age_squared":      tire_age_squared,
        "rolling_avg_time":      inp["rolling_avg"],
        "degradation_rate":      inp["degradation"],
        "gap_ahead":             inp["gap_ahead"],
        "gap_behind":            inp["gap_behind"],
        "position":              inp["position"],
        "race_completion":       race_completion,
        "undercut_threat":       undercut_threat,
        "circuit_id":            inp["circuit_id"],
        "pit_loss_time":         inp["pit_loss_time"],
        "track_temp":            inp["track_temp"],
        "rainfall":              inp["rainfall"],
        "ahead_compound":        inp["ahead_compound"],
        "ahead_tire_age":        inp["ahead_tire_age"],
        "behind_compound":       inp["behind_compound"],
        "behind_tire_age":       bta,
        "expected_stint_length": expected_sl,
    }
    return features, race_completion, tire_age_squared, undercut_threat, expected_sl


def print_result(inp, features, race_completion, tire_age_sq,
                 undercut_threat, expected_sl, probability):
    header("PREDICTION RESULT")

    section("Circuit")
    row("Name",      inp["circuit_name"])
    row("Circuit ID", inp["circuit_id"])
    row("Pit Loss",  f"{inp['pit_loss_time']:+.2f}", "s")

    section("Race Context")
    row("Lap",         f"{inp['current_lap']} / {inp['total_laps']}")
    row("Race compl.", f"{race_completion*100:.1f}", "%")
    row("Position",    inp["position"])

    section("Tire State")
    row("Compound",         COMPOUND_NAMES[inp["compound"]])
    row("Tire age",         f"{inp['tire_age']:.0f}", "laps")
    row("Tire age²",        f"{tire_age_sq:.0f}")
    row("Rolling avg",      f"{inp['rolling_avg']:.3f}", "s")
    row("Degradation rate", f"{inp['degradation']:+.3f}", "s")
    row("Expected stint",   f"{expected_sl:.1f}", "laps")
    pct_through = 100 * inp["tire_age"] / expected_sl if expected_sl > 0 else 0
    row("Stint progress",   f"{pct_through:.0f}", "% of expected")

    section("Gap & Threat")
    row("Gap ahead",       f"{inp['gap_ahead']:.3f}" if inp["gap_ahead"] < 999 else "LEADING")
    row("Gap behind",      f"{inp['gap_behind']:.3f}" if inp["gap_behind"] < 999 else "LAST")
    row("Undercut threat", f"{undercut_threat:.4f}")

    section("Opponent State")
    ahead_cmp_str = COMPOUND_NAMES.get(inp["ahead_compound"], "N/A")
    behind_cmp_str = COMPOUND_NAMES.get(inp["behind_compound"], "N/A")
    if inp["ahead_compound"] == -1:
        row("Car ahead",   "LEADING (no car ahead)")
    else:
        row("Ahead compound",  ahead_cmp_str)
        row("Ahead tire age",  f"{inp['ahead_tire_age']:.0f}", "laps")
    if inp["behind_compound"] == -1:
        row("Car behind",  "LAST (no car behind)")
    else:
        row("Behind compound", behind_cmp_str)
        row("Behind tire age", f"{inp['behind_tire_age']:.0f}", "laps")

    section("Weather")
    row("Track temp", f"{inp['track_temp']:.1f}", "°C")
    row("Rainfall",   "WET 🌧" if inp["rainfall"] == 1 else "DRY ☀")

    # ── Recommendation ────────────────────────────────────────────────────────
    pct = probability * 100
    pit = probability >= THRESHOLD

    if probability >= 0.80:
        confidence_label = c("HIGH", RED, BOLD)
        confidence_note  = "Strong signal — model is confident"
    elif probability >= THRESHOLD:
        confidence_label = c("MEDIUM", YELLOW, BOLD)
        confidence_note  = "Above threshold — lean towards pitting"
    elif probability >= 0.40:
        confidence_label = c("LOW", YELLOW)
        confidence_note  = "Borderline — monitor next 2–3 laps"
    else:
        confidence_label = c("NONE", DIM)
        confidence_note  = "No significant pit signal"

    print()
    print(c("  " + "═" * 56, CYAN))

    rec_text = c("  🟢  PIT NOW", GREEN, BOLD) if pit else c("  🔴  STAY OUT", RED, BOLD)
    print(rec_text)
    print(c(f"\n  Pit probability  :  {pct:.1f}%", WHITE, BOLD))
    print(c(f"  Threshold        :  {THRESHOLD*100:.1f}%  (F1-maximising)", DIM))
    print(c(f"  Confidence       :  ", DIM) + confidence_label)
    print(c(f"  Note             :  {confidence_note}", DIM))

    # Probability bar
    bar_total  = 40
    filled     = int(round(bar_total * probability))
    thresh_pos = int(round(bar_total * THRESHOLD))
    bar_chars  = list("─" * bar_total)
    bar_chars[thresh_pos] = "│"
    for i in range(filled):
        bar_chars[i] = "▓" if i < thresh_pos and pit else "█"
    bar_str = "".join(bar_chars)
    bar_col = GREEN if pit else YELLOW if probability >= 0.40 else DIM
    print()
    print(c(f"  0%  [{bar_str}]  100%", bar_col))
    print(c(f"       {'↑':>{thresh_pos + 1}}  t={THRESHOLD}", DIM))
    print()
    print(c("  " + "═" * 56, CYAN))
    print()


def run_again():
    ans = input(c("\n  Run another prediction? (y/n): ", CYAN)).strip().lower()
    return ans in ("y", "yes", "")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    try:
        model = xgb.XGBClassifier()
        model.load_model("pit_model_v9.json")
    except Exception as e:
        print(c(f"\n  ✗ Could not load pit_model_v9.json: {e}", RED))
        print(c("  Make sure pit_model_v9.json is in the same directory.\n", DIM))
        sys.exit(1)

    print(c("\n  Model loaded: pit_model_v9.json  ✓", GREEN))
    print(c(f"  Features : 19    Threshold : {THRESHOLD}", DIM))

    while True:
        try:
            inp = collect_inputs()
        except KeyboardInterrupt:
            print(c("\n\n  Exiting. Good luck with the strategy! 🏎️\n", CYAN))
            break

        features, race_completion, tire_age_sq, undercut_threat, expected_sl = \
            build_feature_vector(inp)

        X = np.array([[features[col] for col in FEATURE_COLS]], dtype=float)
        probability = float(model.predict_proba(X)[0, 1])

        print_result(inp, features, race_completion, tire_age_sq,
                     undercut_threat, expected_sl, probability)

        if not run_again():
            print(c("\n  Exiting. Good luck with the strategy! 🏎️\n", CYAN))
            break


if __name__ == "__main__":
    main()
