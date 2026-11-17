# Tuning plan (pre-registered)

This file is committed alone, before the first trial. Any change after results needs a new ADR that says it was made after seeing results.

## Tool and budget
- Optuna, TPE sampler, seed 42, no pruner.
- 30 trials or 45 minutes of wall time, whichever comes first. A trial running at the time limit finishes. No extra trials after seeing results.

## Data
- One sample of 300,000 trips, drawn with seed 42 from training days Jan 1 to 24, after the brief's scope filters. The same sample is used for the baseline and for every trial.

## Cross-validation
3 time-ordered expanding folds by pickup day. The same folds for the baseline and every trial.

| Fold | Train (pickup days) | Validate (pickup days) |
|---|---|---|
| 1 | Jan 1 to 12 | Jan 13 to 16 |
| 2 | Jan 1 to 16 | Jan 17 to 20 |
| 3 | Jan 1 to 20 | Jan 21 to 24 |

A 60-minute embargo sits before each validation window: training trips must end before it starts.

## Metric
RMSE in minutes, mean over the 3 folds. MAE is reported only.

## Baseline
- RandomForestRegressor with default settings, random_state=42, plus the training-mean reference.
- Its CV RMSE is locked before tuning starts.

## Search space
| Parameter | Range |
|---|---|
| n_estimators | 50 to 300 |
| max_depth | 8 to 30 |
| min_samples_leaf | 1 to 50, log scale |
| max_features | one of sqrt, 0.3, 0.5, 1.0 |
| max_samples | 0.3 to 1.0 |

## Candidates and limits
- Candidates are the baseline plus every completed trial.
- Each candidate records two numbers from its fold-3 model: the skops file size, and the median single-row prediction time.
- Prediction time is measured on this machine: load the model once, make one warm-up call, then take the median of 100 single-row predictions.
- Limits: file size under 300 MB, and one prediction under 50 ms.

## Decision rule
Apply these steps in order:
1. A tuned candidate is eligible only if its mean CV RMSE is at least 2% lower than the baseline's, it wins on every fold, and it passes both limits. If any tuned candidate is eligible, the one with the best mean CV RMSE is chosen.
2. Otherwise the baseline is kept, if it passes both limits.
3. If the baseline fails a limit, the candidate with the best mean CV RMSE among those that pass both limits is deployed, and the README says so.

The chosen candidate is then retrained on the full training sample (the same 300,000 trips). The two limits are measured again on the retrained model. If it fails a limit, the next-best passing candidate, ordered by mean CV RMSE, is retrained and checked the same way.

## Test week
- Jan 25 to 31 is never used for tuning or choosing.
- The retrained model is scored once on the full test week, after the scope filters.
- The baseline is scored as well, and both scores are reported. The test score never changes the choice.

## Changes
Any change after results needs a new ADR that says it was made after seeing results.
