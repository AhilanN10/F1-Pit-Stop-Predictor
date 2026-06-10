#!/usr/bin/env python3
"""
F1 Pit Stop Predictor — powered by pit_model_v8.json
======================================================
Interactive CLI that takes a current race state as input and outputs
a pit stop recommendation using the trained XGBoost v8 model
(Optuna-tuned, 14 features including track_temp and rainfall).

Usage:
    python3 pit_predictor.py
"""

import sys
import numpy as np
import xgboost as xgb

# ── Constants ─────────────────────────────────────────────────────────────────

THRESHOLD = 0.65    # F1-maximising threshold from v8 PR curve

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

COMPOUND_NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}

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
    header("F1 PIT STOP PREDICTOR  ·  v8 MODEL")

    print(c("\n  Enter the current race state below.", DIM))
    print(c("  Type 999 for gap fields when leading / last place.\n", DIM))

    circuit_id     = choose_circuit()
    circuit_name   = CIRCUIT_NAMES[circuit_id]
    pit_loss_time  = PIT_LOSS_TIME[circuit_id]

    section("Lap & Race State")
    current_lap    = prompt("Current lap number", int, min_=1)
    total_laps     = prompt("Total race laps", int, min_=current_lap)
    position       = prompt("Current position (1–20)", int, min_=1, max_=20)

    section("Tire State")
    compound       = prompt("Compound  (0=Soft  1=Medium  2=Hard)", int,
                            choices=[0, 1, 2])
    tire_age       = prompt("Tire age (laps on this set)", float, min_=0)

    section("Pace & Degradation")
    rolling_avg    = prompt("3-lap rolling avg lap time (seconds)", float, min_=60)
    degradation    = prompt("Degradation rate (rolling_avg - stint base, can be negative)", float)

    section("Gap Information")
    gap_ahead      = prompt("Gap to car ahead (seconds, 999 if leading)", float, min_=0)
    gap_behind     = prompt("Gap to car behind (seconds, 999 if last)", float, min_=0)

    section("Undercut Threat")
    if gap_behind >= 999.0:
        print(c("    (No car behind — undercut_threat will be 0)", DIM))
        behind_tire_age = 0.0
    else:
        behind_tire_age = prompt("Behind car's tire age (laps)", float, min_=0)

    section("Weather Conditions")
    track_temp = prompt("Track surface temperature (°C)", float, min_=-10, max_=80)
    rainfall   = prompt("Rainfall  (0 = dry  1 = wet)", int, choices=[0, 1])

    return {
        "circuit_id":      circuit_id,
        "circuit_name":    circuit_name,
        "pit_loss_time":   pit_loss_time,
        "current_lap":     current_lap,
        "total_laps":      total_laps,
        "position":        position,
        "compound":        compound,
        "tire_age":        tire_age,
        "rolling_avg":     rolling_avg,
        "degradation":     degradation,
        "gap_ahead":       gap_ahead,
        "gap_behind":      gap_behind,
        "behind_tire_age": behind_tire_age,
        "track_temp":      track_temp,
        "rainfall":        rainfall,
    }


def build_feature_vector(inp):
    """Compute derived features and return ordered feature array."""
    race_completion  = inp["current_lap"] / inp["total_laps"]
    tire_age_squared = inp["tire_age"] ** 2

    if inp["gap_behind"] >= 999.0:
        undercut_threat = 0.0
    else:
        denom = inp["tire_age"] - inp["behind_tire_age"] + 1
        undercut_threat = 0.0 if denom <= 0 else inp["gap_behind"] / denom

    features = {
        "compound":         inp["compound"],
        "tire_age":         inp["tire_age"],
        "tire_age_squared": tire_age_squared,
        "rolling_avg_time": inp["rolling_avg"],
        "degradation_rate": inp["degradation"],
        "gap_ahead":        inp["gap_ahead"],
        "gap_behind":       inp["gap_behind"],
        "position":         inp["position"],
        "race_completion":  race_completion,
        "undercut_threat":  undercut_threat,
        "circuit_id":       inp["circuit_id"],
        "pit_loss_time":    inp["pit_loss_time"],
        "track_temp":       inp["track_temp"],
        "rainfall":         inp["rainfall"],
    }
    return features, race_completion, tire_age_squared, undercut_threat


def print_result(inp, features, race_completion, tire_age_sq, undercut_threat, probability):
    header("PREDICTION RESULT")

    section("Circuit")
    row("Name",        inp["circuit_name"])
    row("Circuit ID",  inp["circuit_id"])
    row("Pit Loss",    f"{inp['pit_loss_time']:+.2f}", "s")

    section("Race Context")
    row("Lap",          f"{inp['current_lap']} / {inp['total_laps']}")
    row("Race compl.",  f"{race_completion*100:.1f}", "%")
    row("Position",     inp["position"])

    section("Tire State")
    row("Compound",         COMPOUND_NAMES[inp["compound"]])
    row("Tire age",         f"{inp['tire_age']:.0f}", "laps")
    row("Tire age²",        f"{tire_age_sq:.0f}")
    row("Rolling avg",      f"{inp['rolling_avg']:.3f}", "s")
    row("Degradation rate", f"{inp['degradation']:+.3f}", "s")

    section("Gap & Threat")
    row("Gap ahead",        f"{inp['gap_ahead']:.3f}" if inp["gap_ahead"] < 999 else "LEADING")
    row("Gap behind",       f"{inp['gap_behind']:.3f}" if inp["gap_behind"] < 999 else "LAST")
    row("Undercut threat",  f"{undercut_threat:.4f}")

    section("Weather")
    row("Track temp",       f"{inp['track_temp']:.1f}", "°C")
    row("Rainfall",         "WET 🌧" if inp["rainfall"] == 1 else "DRY ☀")

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

    if pit:
        rec_text = c("  🟢  PIT NOW", GREEN, BOLD)
    else:
        rec_text = c("  🔴  STAY OUT", RED, BOLD)

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
    bar_chars[thresh_pos] = "│"          # threshold marker
    for i in range(filled):
        if i < thresh_pos:
            bar_chars[i] = "█" if not pit else "▓"
        else:
            bar_chars[i] = "█"
    bar_str   = "".join(bar_chars)
    bar_col   = GREEN if pit else YELLOW if probability >= 0.40 else DIM
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
    # Load model once
    try:
        model = xgb.XGBClassifier()
        model.load_model("pit_model_v8.json")
    except Exception as e:
        print(c(f"\n  ✗ Could not load pit_model_v8.json: {e}", RED))
        print(c("  Make sure pit_model_v8.json is in the same directory.\n", DIM))
        sys.exit(1)

    print(c("\n  Model loaded: pit_model_v8.json  ✓", GREEN))
    print(c(f"  Features : 14    Threshold : {THRESHOLD}", DIM))

    while True:
        try:
            inp = collect_inputs()
        except KeyboardInterrupt:
            print(c("\n\n  Exiting. Good luck with the strategy! 🏎️\n", CYAN))
            break

        features, race_completion, tire_age_sq, undercut_threat = build_feature_vector(inp)

        # Build ordered array matching training feature order
        X = np.array([[features[col] for col in FEATURE_COLS]], dtype=float)
        probability = float(model.predict_proba(X)[0, 1])

        print_result(inp, features, race_completion, tire_age_sq, undercut_threat, probability)

        if not run_again():
            print(c("\n  Exiting. Good luck with the strategy! 🏎️\n", CYAN))
            break


if __name__ == "__main__":
    main()
