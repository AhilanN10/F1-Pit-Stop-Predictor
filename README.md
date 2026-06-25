# F1 Pit Stop Predictor
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-pit-stop-predictor.streamlit.app/)
An ensemble machine learning system that predicts optimal pit stop windows in Formula 1 racing using real telemetry and timing data from the 2021 and 2023 seasons.

## Overview
This project builds an ensemble prediction system combining an XGBoost classifier and a 2-layer LSTM neural network to predict F1 pit stop windows. The system is trained on 42,572 laps of real F1 data from the 2021 and 2023 seasons pulled from the FastF1 API. The ensemble achieves ROC-AUC 0.9289, Average Precision 0.3523, and 9.6x lift over the random baseline.

## Architecture
The system uses two complementary models whose probability outputs are averaged with equal 0.5 weights.

**XGBoost Classifier — `pit_model_v9.json`**
Trained on individual lap snapshots with 19 engineered features. Hyperparameters optimized with Optuna (100 trials, 5-fold stratified CV, Average Precision objective). Achieves AP 0.1689 and ROC-AUC 0.8363. Strong at identifying high-confidence pit events from a single-lap view with sharp precision at the top of the probability range.

**2-Layer LSTM — `pit_model_lstm.pt`**
Trained on sliding windows of 5 consecutive laps per driver per race (14,927 sequences from 2023). Hidden size 64, dropout 0.3, BCEWithLogitsLoss with class-ratio pos_weight. Achieves AP 0.1488 and ROC-AUC 0.8478. Captures temporal degradation trends and positional dynamics that are invisible to a single-lap model.

**Ensemble**
Final probability = 0.5 × p(XGBoost) + 0.5 × p(LSTM). Achieves AP 0.3523 and ROC-AUC 0.9289, more than doubling the AP of either individual model. The LSTM reduces missed pit stops (low false negatives); the XGBoost reduces false alarms at high thresholds. Together they achieve 94 true positives out of 110 pit events at the default threshold of 0.50.

## Features
The system uses 19 engineered features per lap. `expected_stint_length` is the single most important feature at 14.4% XGBoost importance.

| Feature | Description |
|---|---|
| compound | Tire compound: Soft=0, Medium=1, Hard=2 |
| tire_age | Laps on current tire set |
| tire_age_squared | Quadratic term to capture nonlinear degradation cliff |
| rolling_avg_time | 3-lap rolling average lap time in seconds |
| degradation_rate | Rolling average minus stint baseline; measures pace loss |
| gap_ahead | Gap to car ahead in seconds, 999 if leading |
| gap_behind | Gap to car behind in seconds, 999 if last |
| position | Current race position |
| race_completion | Lap / total laps, range 0.0 to 1.0 |
| undercut_threat | gap_behind ÷ (own_tire_age − behind_tire_age + 1) |
| circuit_id | Stable alphabetical integer encoding across seasons, 0 to 27 |
| pit_loss_time | Cross-season mean seconds lost in pit lane at this circuit |
| track_temp | Track surface temperature in °C from FastF1 weather data |
| rainfall | Binary: 1 if any rain recorded during that lap, 0 otherwise |
| ahead_compound | Compound of car directly ahead (same encoding, −1 if leading) |
| ahead_tire_age | Tire age of car directly ahead in laps (−1 if leading) |
| behind_compound | Compound of car directly behind (same encoding, −1 if last) |
| behind_tire_age | Tire age of car directly behind in laps (−1 if last) |
| expected_stint_length | Cross-season mean stint length in laps for this compound at this circuit |

## Model Performance

| Version | ROC-AUC | AP | Lift | Features | Notes |
|---|:-:|:-:|:-:|:-:|---|
| v4 | 0.8267 | 0.1446 | 4.7× | 12 | XGBoost baseline |
| v8 | 0.8393 | 0.1545 | 5.0× | 14 | Added weather + Optuna tuning |
| v9 | 0.8363 | 0.1689 | 5.5× | 19 | Added opponent context + stint length |
| LSTM | 0.8478 | 0.1488 | — | 19 | 2-layer LSTM on 5-lap sequences |
| **Ensemble** | **0.9289** | **0.3523** | **9.6×** | **19** | **XGBoost v9 + LSTM, best F1 0.4567 at t=0.735** |

### Ensemble Threshold Table

| Threshold | Precision | Recall | F1 | Notes |
|:-:|:-:|:-:|:-:|---|
| 0.40 | 16.2% | 90.0% | 0.274 | Maximum recall |
| 0.50 | 19.9% | 85.5% | 0.323 | Default |
| 0.60 | 24.7% | 75.5% | 0.372 | |
| **0.735** | **~33%** | **61.8%** | **0.457** | **Best F1 — recommended operating threshold** |

## Project Structure
```
f1_data_pipeline.py      Data pipeline: FastF1 → 26-column CSV with all 19 features
prepare_data.py          Legacy data preparation script
train_model_v4.py        XGBoost v4 baseline (12 features)
train_model_v8.py        XGBoost v8 with weather features and Optuna (14 features)
train_model_v9.py        XGBoost v9 with opponent context and stint length (19 features)
train_model_lstm.py      2-layer LSTM on 5-lap sequences (19 features)
train_model_ensemble.py  Ensemble evaluation: XGBoost v9 + LSTM, equal weights
pit_predictor.py         Interactive CLI predictor (v9 model, 19 features)
pit_model_v4.json        Trained XGBoost v4 model
pit_model_v9.json        Trained XGBoost v9 model
pit_model_lstm.pt        Trained LSTM model + StandardScaler state dict
f1_pit_data_2021_2023.csv  Processed dataset (42,572 laps, 26 columns)
```

## Data Pipeline
Data is pulled from the FastF1 Python library which interfaces with F1's official timing feed. The pipeline covers the 2021 and 2023 seasons (42,572 laps across 28 unique circuits). 2022 is excluded due to a FastF1 internal issue that corrupts session state on that season's data.

Circuit IDs are assigned alphabetically across all seasons (0 to 27) for stable cross-season encoding. Pit loss time and expected stint length are derived empirically as cross-season means grouped by circuit and compound. Weather data (track temperature and rainfall) is merged via `pd.merge_asof` with backward direction to match the nearest weather reading to each lap timestamp.

Filters applied before training:
- Safety car and virtual safety car laps are dropped since they distort tire degradation and pit timing signals
- The first two laps of every stint are dropped since the 3-lap rolling average requires at least 3 data points
- Only Soft, Medium, and Hard compounds are retained

## Validation
The system was validated against a known strategic moment: Max Verstappen at the 2023 Bahrain Grand Prix, lap 14 of 57, leading on 14-lap-old Soft tires with Perez 1.8 seconds behind on similar-age tyres. The XGBoost v9 model returned a PIT NOW recommendation at 72.4% probability, correctly identifying the undercut threat that drove Red Bull's actual pit call on that lap.

## Known Limitations
Precision at the operating threshold is around 13–15% for the individual XGBoost and LSTM models, meaning roughly 1 in 7 pit alerts is correct. This improves significantly in the ensemble due to the complementary error profiles of the two architectures.

The LSTM requires a 5-lap buffer before it can generate predictions. It cannot make predictions on the first 4 laps of any stint. In a live deployment, the ensemble would fall back to XGBoost-only probability for those early laps.

The models are trained on 2021 and 2023 data. Regulation changes (2022 ground-effect cars, 2026 power unit rules), new circuits, or tyre compound reformulations in other seasons may reduce accuracy.

## Usage
Install dependencies:
```
pip install fastf1 xgboost scikit-learn pandas numpy torch optuna matplotlib
```

Run the interactive predictor:
```
python3 pit_predictor.py
```

Regenerate the dataset from FastF1 cache:
```
python3 f1_data_pipeline.py
```

Retrain XGBoost v9:
```
python3 train_model_v9.py
```

Retrain LSTM:
```
python3 train_model_lstm.py
```

Evaluate ensemble:
```
python3 train_model_ensemble.py
```

## Dependencies
- **FastF1** — telemetry and timing data from the official F1 feed
- **XGBoost** — gradient boosted classifier
- **PyTorch** — 2-layer LSTM sequence model
- **scikit-learn** — evaluation metrics, data splitting, StandardScaler
- **Optuna** — hyperparameter optimisation for XGBoost v9
- **pandas** and **numpy** — data processing
- **matplotlib** — precision-recall curve visualisations
