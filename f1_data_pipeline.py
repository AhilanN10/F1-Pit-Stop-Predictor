"""
F1 Pit Stop Prediction Data Pipeline
=====================================
Pulls race lap data for every Grand Prix from the 2023 F1 season
using the FastF1 library and builds a flat feature dataframe for ML.

Features extracted per lap per driver:
  - compound          : tire compound (Soft=0, Medium=1, Hard=2)
  - tire_age          : age of current tire set in laps
  - tire_age_squared  : tire_age²  (captures non-linear degradation curve)
  - rolling_avg_time  : 3-lap rolling average lap time (seconds)
  - degradation_rate  : rolling_avg_time minus average of first 3 stint laps
  - gap_ahead         : gap to car ahead in seconds
  - gap_behind        : gap to car behind in seconds
  - position          : current race position
  - race_completion   : fraction of race completed (0.0–1.0)
  - undercut_threat   : gap_behind / (own_tire_age - behind_tire_age + 1)
  - circuit_id        : circuit encoded as 0-based integer in calendar order
  - pit_loss_time     : mean lap-time penalty for pitting at this circuit (seconds)
  - pitted_next_lap   : 1 if driver pitted on next lap, else 0  [LABEL]

Filters applied:
  - Drop laps run under Safety Car / Virtual Safety Car periods
  - Drop the final lap of any stint where the driver retired
    (i.e., did not pit and did not reach the total race lap count)
"""

import logging
import subprocess
import sys
import time
import warnings

subprocess.run([sys.executable, "-m", "pip", "install", "fastf1"], check=True)

import fastf1
from fastf1.exceptions import RateLimitExceededError
import numpy as np
import pandas as pd
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── FastF1 cache ───────────────────────────────────────────────────────────────
CACHE_DIR = Path("./fastf1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Constants ──────────────────────────────────────────────────────────────────
SEASONS = range(2023, 2024)      # 2023 only
OUTPUT_CSV = "f1_pit_data_2023.csv"

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

# FastF1 / FIA TrackStatus codes that indicate SC / VSC deployment.
# TrackStatus is a string; each character is an active code.
#   1 = AllClear  2 = Yellow  3 = ?  4 = SC  5 = Red  6 = VSC  7 = VSC Ending
SC_STATUS_CHARS = {"4", "6", "7"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _td_to_s(td) -> float:
    """Convert a pandas Timedelta (or NaT) to float seconds."""
    if pd.isnull(td):
        return np.nan
    return td.total_seconds()


# ── Rate-limit retry ───────────────────────────────────────────────────────────
_RATE_LIMIT_WINDOW = 3600   # FastF1 enforces a 500-calls/hour rolling window
_RATE_LIMIT_BUFFER = 120    # extra seconds to wait after the window resets

def _load_with_retry(session, max_retries: int = 3, **load_kwargs):
    """
    Call session.load(**load_kwargs), retrying after a rate-limit pause.

    FastF1 raises RateLimitExceededError when the 500 calls/hour quota is
    exhausted.  We sleep for the remainder of the rolling 1-hour window
    (plus a small buffer) and try again up to `max_retries` times.
    """
    for attempt in range(1, max_retries + 1):
        try:
            session.load(**load_kwargs)
            return
        except RateLimitExceededError as exc:
            if attempt == max_retries:
                raise
            wait = _RATE_LIMIT_WINDOW + _RATE_LIMIT_BUFFER
            log.warning(
                "  Rate limit hit (attempt %d/%d). Sleeping %d s (~%.0f min) …",
                attempt, max_retries, wait, wait / 60,
            )
            time.sleep(wait)


def _get_schedule_with_retry(year: int, max_retries: int = 3):
    """Fetch the event schedule, retrying on rate-limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return fastf1.get_event_schedule(year, include_testing=False)
        except RateLimitExceededError:
            if attempt == max_retries:
                raise
            wait = _RATE_LIMIT_WINDOW + _RATE_LIMIT_BUFFER
            log.warning(
                "  Rate limit fetching schedule (attempt %d/%d). Sleeping %d s …",
                attempt, max_retries, wait,
            )
            time.sleep(wait)


def _rolling_mean(series: pd.Series, window: int = 3) -> pd.Series:
    """Rolling mean requiring exactly `window` data points."""
    return series.rolling(window=window, min_periods=window).mean()


def _stint_base_time(lap_times: pd.Series) -> float:
    """Mean of the first 3 valid lap times in a stint."""
    valid = lap_times.dropna()
    return valid.iloc[:3].mean() if len(valid) >= 3 else np.nan


def _is_sc_lap(status: str) -> bool:
    """Return True if a TrackStatus string contains any SC/VSC code."""
    if not isinstance(status, str):
        return False
    return bool(SC_STATUS_CHARS.intersection(set(status)))


# ──────────────────────────────────────────────────────────────────────────────
# Gap computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add `gap_ahead_s` and `gap_behind_s` columns (seconds) to the laps
    dataframe by comparing each driver's session-time at lap-end (`Time`).

    Strategy
    --------
    For lap N, driver A has completed N laps at session-time T_A.
    If driver B is one position ahead (fewer laps OR same laps but lower time),
    the gap is |T_A - T_B|.

    FastF1's `Time` column is the session clock at which the lap ENDED.
    For drivers on the same lap count this is the correct on-track gap.
    When drivers are on different lap counts the gap is not strictly meaningful
    but is kept as NaN rather than producing a misleading number.
    """
    laps = laps.copy()
    laps["gap_ahead_s"] = np.nan
    laps["gap_behind_s"] = np.nan

    # We need: LapNumber, Position, Time (session time at lap end), Driver
    required = {"LapNumber", "Position", "Time", "Driver"}
    if not required.issubset(laps.columns):
        return laps

    # Convert Time to seconds once for speed
    laps["_time_s"] = laps["Time"].apply(_td_to_s)

    for lap_num, grp in laps.groupby("LapNumber"):
        grp = grp.dropna(subset=["Position", "_time_s"]).sort_values("Position")
        if len(grp) < 2:
            continue

        idxs = grp.index.tolist()
        times = grp["_time_s"].values  # sorted by position ascending

        # gap_ahead[i] = times[i] - times[i-1]   (car ahead finished the lap earlier)
        # gap_behind[i] = times[i+1] - times[i]
        for rank, idx in enumerate(idxs):
            if rank > 0:
                laps.at[idx, "gap_ahead_s"] = times[rank] - times[rank - 1]
            if rank < len(idxs) - 1:
                laps.at[idx, "gap_behind_s"] = times[rank + 1] - times[rank]

    laps.drop(columns=["_time_s"], inplace=True)
    return laps


# ──────────────────────────────────────────────────────────────────────────────
# Undercut threat computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_undercut_threat(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `undercut_threat` column to the all-driver laps dataframe.

    For each (lap, driver) row:
      - Find the car directly behind (Position + 1) on the same lap.
      - Look up that car's TyreLife as `behind_tire_age`.
      - Compute:  undercut_threat = gap_behind_s / (own_tire_age - behind_tire_age + 1)
      - If gap_behind_s is NaN or 999.0 (last place / no car behind), set to 0.

    The denominator is shifted by +1 to avoid division by zero and to dampen
    the signal when tyre ages are similar.  A higher value means the car
    behind has much fresher tyres AND is close — classic undercut setup.

    Requires columns: Position, TyreLife, gap_behind_s, LapNumber.
    """
    laps = laps.copy()
    laps["undercut_threat"] = 0.0

    required = {"LapNumber", "Position", "TyreLife", "gap_behind_s"}
    if not required.issubset(laps.columns):
        return laps

    for lap_num, grp in laps.groupby("LapNumber"):
        grp = grp.dropna(subset=["Position"]).sort_values("Position")
        if len(grp) < 2:
            continue

        # Build a position → (index, TyreLife) lookup for this lap
        pos_to_idx       = dict(zip(grp["Position"], grp.index))
        pos_to_tyre_age  = dict(zip(grp["Position"],
                                    grp["TyreLife"].astype(float)))

        for idx, row in grp.iterrows():
            own_pos      = row["Position"]
            gap_behind   = row["gap_behind_s"]
            own_tire_age = pos_to_tyre_age.get(own_pos, np.nan)
            behind_pos   = own_pos + 1

            # No car behind, or last place sentinel — threat is zero
            if pd.isna(gap_behind) or gap_behind >= 999.0:
                laps.at[idx, "undercut_threat"] = 0.0
                continue

            behind_tire_age = pos_to_tyre_age.get(behind_pos, np.nan)

            if pd.isna(own_tire_age) or pd.isna(behind_tire_age):
                # Can't compute — leave as 0
                continue

            denominator = own_tire_age - behind_tire_age + 1
            if denominator <= 0:
                # Car behind has older or equal-age tyres — no undercut incentive
                laps.at[idx, "undercut_threat"] = 0.0
                continue
            laps.at[idx, "undercut_threat"] = gap_behind / denominator

    return laps


# ──────────────────────────────────────────────────────────────────────────────
# Per-driver feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_driver_features(driver_laps: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """
    Build lap-level features for a single driver in a race.

    Parameters
    ----------
    driver_laps : filtered + sorted laps for one driver (already SC-filtered)
    total_laps  : total race laps (denominator for race_completion)

    Returns
    -------
    pd.DataFrame with all feature columns, or empty DataFrame.
    """
    if driver_laps.empty:
        return pd.DataFrame()

    df = driver_laps.copy().reset_index(drop=True)

    # Lap time → seconds
    df["lap_time_s"] = df["LapTime"].apply(_td_to_s)

    # Race completion
    df["race_completion"] = df["LapNumber"] / total_laps

    # Compound encoding
    df["compound"] = df["Compound"].str.upper().map(COMPOUND_MAP)

    # ── Stint-level features ─────────────────────────────────────────────────
    stint_chunks = []
    for _, stint_df in df.groupby("Stint", sort=True):
        stint_df = stint_df.sort_values("LapNumber").reset_index(drop=True)
        lap_times = stint_df["lap_time_s"]

        # Tire age: use TyreLife if available
        if "TyreLife" in stint_df.columns and stint_df["TyreLife"].notna().any():
            tire_age = stint_df["TyreLife"].astype(float)
        else:
            tire_age = pd.Series(range(1, len(stint_df) + 1), dtype=float)

        rolling_avg = _rolling_mean(lap_times, window=3)
        base_time = _stint_base_time(lap_times)
        degradation = rolling_avg - base_time

        stint_df = stint_df.copy()
        stint_df["tire_age"]         = tire_age.values
        stint_df["tire_age_squared"] = (tire_age ** 2).values
        stint_df["rolling_avg_time"] = rolling_avg.values
        stint_df["degradation_rate"] = degradation.values
        stint_chunks.append(stint_df)

    if not stint_chunks:
        return pd.DataFrame()

    df = (
        pd.concat(stint_chunks, ignore_index=True)
        .sort_values("LapNumber")
        .reset_index(drop=True)
    )

    # ── pitted_next_lap label ────────────────────────────────────────────────
    # PitInTime is non-null on the lap where the driver entered the pits.
    pit_flag = df["PitInTime"].notna().astype(int)
    df["pitted_next_lap"] = pit_flag.shift(-1).fillna(0).astype(int)

    # ── Drop retirement final lap ────────────────────────────────────────────
    # If the driver's last lap is below total_laps AND they didn't pit on that lap,
    # they retired — drop that orphan row so we don't mis-label it as pitted=0.
    last_idx = df.index[-1]
    last_lap_num = df.loc[last_idx, "LapNumber"]
    last_pit_in = df.loc[last_idx, "PitInTime"]
    if last_lap_num < total_laps and pd.isnull(last_pit_in):
        df = df.drop(index=last_idx).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Race-level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_race(session) -> pd.DataFrame:
    """
    Load and process one race session.
    Returns a feature DataFrame or an empty DataFrame on failure/no data.
    """
    try:
        _load_with_retry(session, laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        log.warning("  Could not load session: %s", exc)
        return pd.DataFrame()

    laps = session.laps
    if laps is None or laps.empty:
        log.warning("  No lap data found.")
        return pd.DataFrame()

    # ── Drop SC / VSC laps ────────────────────────────────────────────────────
    if "TrackStatus" in laps.columns:
        sc_mask = laps["TrackStatus"].apply(_is_sc_lap)
        laps = laps[~sc_mask].copy()

    if laps.empty:
        return pd.DataFrame()

    # ── Total laps (use session max, not SC-filtered max) ─────────────────────
    total_laps = int(session.laps["LapNumber"].max())
    if total_laps < 1:
        return pd.DataFrame()

    # ── Keep only known dry compounds ────────────────────────────────────────
    laps = laps[laps["Compound"].str.upper().isin(COMPOUND_MAP)].copy()
    if laps.empty:
        return pd.DataFrame()

    # ── Gap computation using session-time arithmetic ─────────────────────────
    laps = compute_gaps(laps)

    # ── Undercut threat ───────────────────────────────────────────────────────
    laps = compute_undercut_threat(laps)

    # ── Per-driver feature extraction ─────────────────────────────────────────
    year = session.event["EventDate"].year
    round_num = int(session.event["RoundNumber"])
    event_name = session.event["EventName"]

    all_driver_dfs = []
    for drv, drv_laps in laps.groupby("Driver"):
        drv_laps = drv_laps.sort_values("LapNumber")
        feat_df = extract_driver_features(drv_laps, total_laps)
        if feat_df.empty:
            continue

        feat_df["driver"]     = drv
        feat_df["year"]       = year
        feat_df["round"]      = round_num
        feat_df["event_name"] = event_name
        feat_df["circuit_id"] = round_num - 1   # 0-based calendar-order encoding
        all_driver_dfs.append(feat_df)

    if not all_driver_dfs:
        return pd.DataFrame()

    race_df = pd.concat(all_driver_dfs, ignore_index=True)

    # ── Circuit-level pit loss time ─────────────────────────────────────────
    # For each pit-trigger lap N (pitted_next_lap == 1), the NEXT lap for that
    # driver is the slow in-lap (it includes pit-lane traversal in its lap time).
    # pit_loss_raw = lap_time_s(N+1) - rolling_avg_time(N)
    # We average all such values across the race to get a circuit-level constant.
    pit_losses: list[float] = []
    for drv, drv_df in race_df.groupby("driver"):
        drv_df = drv_df.sort_values("LapNumber").reset_index(drop=True)
        pit_rows = drv_df[drv_df["pitted_next_lap"] == 1]
        for _, row in pit_rows.iterrows():
            ref_pace = row.get("rolling_avg_time", np.nan)
            if pd.isna(ref_pace):
                continue
            # Next lap in driver’s sequence is the actual pit in-lap
            later_laps = drv_df[drv_df["LapNumber"] > row["LapNumber"]]
            if later_laps.empty:
                continue
            inlap_time = later_laps.iloc[0].get("lap_time_s", np.nan)
            if pd.notna(inlap_time):
                pit_losses.append(float(inlap_time - ref_pace))

    pit_loss_time = float(np.mean(pit_losses)) if pit_losses else np.nan
    race_df["pit_loss_time"] = pit_loss_time

    # ── Select and rename final columns ───────────────────────────────────────
    col_map = {
        "year":             "year",
        "round":            "round",
        "event_name":       "event_name",
        "driver":           "driver",
        "LapNumber":        "lap_number",
        "Position":         "position",
        "compound":         "compound",
        "tire_age":         "tire_age",
        "tire_age_squared": "tire_age_squared",
        "rolling_avg_time": "rolling_avg_time",
        "degradation_rate": "degradation_rate",
        "gap_ahead_s":      "gap_ahead",
        "gap_behind_s":     "gap_behind",
        "race_completion":  "race_completion",
        "undercut_threat":  "undercut_threat",
        "circuit_id":       "circuit_id",
        "pit_loss_time":    "pit_loss_time",
        "pitted_next_lap":  "pitted_next_lap",
    }
    available = {k: v for k, v in col_map.items() if k in race_df.columns}
    race_df = race_df[list(available.keys())].rename(columns=available)

    return race_df


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    all_races = []
    total_errors = 0
    circuit_map: dict[int, str] = {}   # circuit_id → event_name

    for year in SEASONS:
        log.info("=" * 60)
        log.info("Season %d", year)
        log.info("=" * 60)

        try:
            schedule = _get_schedule_with_retry(year)
        except Exception as exc:
            log.error("Could not fetch schedule for %d: %s", year, exc)
            total_errors += 1
            continue

        # Filter to actual race rounds only (exclude pre-season tests)
        race_rounds = schedule[schedule["EventFormat"].notna()]

        for _, event in race_rounds.iterrows():
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]

            log.info("  [%d R%02d] %s", year, round_num, event_name)

            try:
                session = fastf1.get_session(year, round_num, "R")
                race_df = process_race(session)
            except Exception as exc:
                log.warning("    Error: %s", exc)
                total_errors += 1
                continue

            if race_df.empty:
                log.warning("    No usable data — skipping.")
                continue

            # Record the circuit mapping from this race
            if "circuit_id" in race_df.columns and "event_name" in race_df.columns:
                cid  = int(race_df["circuit_id"].iloc[0])
                name = race_df["event_name"].iloc[0]
                circuit_map[cid] = name

            log.info(
                "    → %d rows | %d pit events",
                len(race_df),
                int(race_df["pitted_next_lap"].sum()),
            )
            all_races.append(race_df)

    if not all_races:
        log.error("No data collected. Exiting.")
        return None

    final_df = pd.concat(all_races, ignore_index=True)

    log.info("")
    log.info("Pipeline complete.")
    log.info("  Total lap rows  : %d", len(final_df))
    log.info(
        "  Pit events      : %d  (%.1f%%)",
        int(final_df["pitted_next_lap"].sum()),
        100 * final_df["pitted_next_lap"].mean(),
    )
    log.info("  Errors / skips  : %d", total_errors)
    log.info("  Saving → %s", OUTPUT_CSV)

    final_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Done ✓")

    # ── Print circuit mapping ───────────────────────────────────────────────
    print()
    print("  CIRCUIT ID MAPPING (circuit_id → event_name)")
    print("  " + "-" * 44)
    for cid in sorted(circuit_map):
        print(f"  {cid:>3}  {circuit_map[cid]}")
    print()

    # ── Print pit loss time per circuit ───────────────────────────────────
    if "pit_loss_time" in final_df.columns and "circuit_id" in final_df.columns:
        plt_map = (
            final_df[["circuit_id", "event_name", "pit_loss_time"]]
            .drop_duplicates(subset="circuit_id")
            .sort_values("circuit_id")
        )
        print("  PIT LOSS TIME PER CIRCUIT")
        print("  " + "-" * 50)
        print(f"  {'ID':>3}  {'Event':<32}  {'Pit Loss (s)':>12}")
        print("  " + "-" * 50)
        for _, r in plt_map.iterrows():
            print(f"  {int(r['circuit_id']):>3}  {r['event_name']:<32}  {r['pit_loss_time']:>12.2f}")
        print("  " + "-" * 50)
        print()

    return final_df


if __name__ == "__main__":
    run_pipeline()
