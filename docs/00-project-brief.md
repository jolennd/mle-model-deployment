# Project brief
Kind (ml | pipeline): ml
Problem: Predict NYC yellow taxi trip duration in minutes from the trip details the API receives (pickup and dropoff zones, trip distance, pickup time), trained on January 2025.
Users: Solo project (course deliverable). Predictions are served locally by an API.
Constraints: Everything runs locally; every port is bound to 127.0.0.1. Data is one static file. Trips outside 1 to 60 minutes are out of scope, so predictions for them are not validated. In the data, trip_distance is the measured distance after the trip; at request time it can only be an estimate, so reported RMSE is optimistic for real requests (ADR 2).

## Fill this block when kind is ml
Label and definition: Trip duration in minutes, dropoff minus pickup. Only trips of 1 to 60 minutes are in scope, in training and in the test week. Only pickups on or after 2025-01-01 00:00 and before 2025-02-01 00:00 count. Trips with dropoff before pickup are dropped as data errors. The staging transform applies the date window and the dropoff-before-pickup cleaning. The Python label code applies the 1 to 60 minute scope the same way in training and in the test week. Dropped row counts per reason (date window, dropoff before pickup) are recorded in docs/02-data-sources.md and summarised in one line in the README. A test checks that raw rows equal kept rows plus dropped rows, per reason.
Label horizon (how far ahead of the prediction time the label is measured): 60 minutes cap (same as the scope filter).
Split unit (the group kept whole in one split, for example a customer or a month): Time, by pickup date. Train Jan 1 to 24, test Jan 25 to 31. The test week stays locked until the baseline lock exists.
Embargo: 60 minutes. Training keeps only trips whose dropoff is before 2025-01-25 00:00.
Primary metric: RMSE in minutes. MAE is reported only.
Baseline to beat: Training-mean duration (naive reference). Locked baseline: RandomForestRegressor with default settings, random_state=42. Its CV RMSE is recorded in the baseline lock before the test week is opened.

## Fill this block when kind is pipeline
Source and refresh rate:
Natural key of the source (or none):
Grain of the output (one row per what):
Model shape (aggregate | star | normalized | obt), and why:
Output tables or files:
Acceptance numbers (must reconcile to the source; say whether rows outside the partition count):
ETL or ELT, and why:
First check that fails when the output is wrong:
