# F1 Pit Stop Predictor
A machine learning system that predicts optimal pit stop windows in Formula 1 racing using real telemetry and timing data from the 2023 season.

## Overview
This project builds a binary classifier that answers one question for any lap of any driver: should this car pit on the next lap? The model is trained on 22,137 laps of real F1 data pulled from the FastF1 API, covering all 22 rounds of the 2023 season.

The system accounts for tire degradation, undercut threat from the car behind, circuit-specific pit lane loss time, and race context to produce a probability score and recommendation at a calibrated decision threshold.

## Project Structure
```
f1_data_pipeline.py   - Pulls and processes race data from FastF1 API with caching
prepare_data.py       - Cleans data and prepares train/test splits
train_model_v4.py     - Trains the XGBoost classifier
pit_predictor.py      - Interactive command-line prediction interface
f1_pit_data_2023.csv  - Processed dataset (22,137 laps, 18 columns)
pit_model_v4.json     - Trained XGBoost model
```

## Features
The model uses 12 engineered features per lap:

| Feature | Description |
|---|---|
| compound | Tire compound encoded as Soft=0, Medium=1, Hard=2 |
| tire_age | Laps on current tire set |
| tire_age_squared | Quadratic tire age to capture nonlinear degradation cliff |
| rolling_avg_time | 3-lap rolling average lap time in seconds |
| degradation_rate | Rolling average minus stint baseline, measures pace loss |
| gap_ahead | Gap to car ahead in seconds, 999 if leading |
| gap_behind | Gap to car behind in seconds, 999 if last |
| position | Current race position |
| race_completion | Lap number divided by total laps, range 0.0 to 1.0 |
| undercut_threat | gap_behind divided by tire age delta plus 1, measures undercut risk |
| circuit_id | Circuit encoded as integer in calendar order, 0 to 21 |
| pit_loss_time | Average seconds lost in the pit lane at this circuit, derived from 2023 data |

## Model
- Algorithm: XGBoost binary classifier
- Training data: 2023 F1 season, all 22 rounds, 15,848 training laps
- Class imbalance handling: scale_pos_weight=32 reflecting the 32:1 no-pit to pit ratio
- Decision threshold: 0.645 (F1-maximizing)

Performance:
- ROC-AUC: 0.8267
- Average Precision: 0.1446 (4.7x above random baseline of 0.031)
- Best F1: 0.2143 at threshold 0.645
- Recall at threshold: 44.3%
- Precision at threshold: 13.3%

## Data Pipeline
Data is pulled from the FastF1 Python library which interfaces with F1's official timing feed. The pipeline applies the following filters before training:

- Safety car and virtual safety car laps are dropped since they distort tire degradation and pit timing signals
- The final lap of any stint ending in retirement is dropped since these are not real pit decisions
- The first two laps of every stint are dropped since the 3-lap rolling average requires at least 3 data points

Pit loss time per circuit is derived empirically from the 2023 data by measuring the time delta between a driver's pit lap and their rolling average at that point, grouped by circuit.

## Known Limitations
Precision is 13.3 percent at the operating threshold, meaning roughly 1 in 8 pit alerts is correct. This is a fundamental ceiling imposed by the available features. The model does not have access to pit wall radio communications, real-time tire temperature data, or safety car probability forecasts, all of which F1 strategists use in practice.

Recall is 44.3 percent, meaning the model misses approximately half of actual pit stops. Catching more stops without generating excessive false alarms would require richer data sources.

The model is trained on 2023 data only. Rule changes, tire compounds, or circuit modifications in other seasons may reduce accuracy.

## Validation
The model was validated against a known strategic moment: Max Verstappen at the 2023 Bahrain Grand Prix, lap 14 of 57, leading on 14-lap-old Soft tires with Perez 1.8 seconds behind. The model returned a PIT NOW recommendation at 65.6 percent probability, correctly identifying the undercut threat that drove Red Bull's actual pit call on that lap.

## Usage
Install dependencies:
```
pip install fastf1 xgboost scikit-learn pandas numpy
```

Run the interactive predictor:
```
python3 pit_predictor.py
```

Regenerate the dataset (uses cache after first run):
```
python3 f1_data_pipeline.py
```

Retrain the model:
```
python3 train_model_v4.py
```

## Dependencies
- FastF1 for telemetry and timing data
- XGBoost for the gradient boosted classifier
- scikit-learn for train/test splitting and evaluation metrics
- pandas and numpy for data processing
