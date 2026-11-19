# Project Description

In this project I build, track, and serve a trip-duration model on the January 2025 NYC Yellow Taxi dataset.

## Main Task

Train and locally deploy a Machine Learning model that predicts the duration of New York City yellow taxi trips.

Use:

- `scikit-learn` to train a `RandomForestRegressor`,
- `MLflow` to track experiments locally,
- `FastAPI` to serve predictions through an API,
- a local request client such as `curl` or `requests` to validate the endpoint.

### Stretch Tasks

- Test your data pipeline, model, or API.
- Use `GridSearchCV` or `Optuna` for hyperparameter tuning.
- Compare the tuned model against the baseline and explain whether the extra complexity was worth it.

## Dataset

Yellow Taxi trip records: [NYC TLC trip record data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- Year: `2025`
- Month: `01`
- Target: `duration`

## Workflow

1. Load the January 2025 taxi data and inspect the columns that influence trip duration.
2. Create a clean training dataset and define the target variable.
3. Split the data into train and validation sets.
4. Train a baseline `RandomForestRegressor` model.
5. Evaluate the model with RMSE and log the run to MLflow.
6. Refactor feature preparation and model loading into modules you can reuse from both training code and the API.
7. Build a FastAPI app with a `/predict` endpoint.
8. Run the API locally and send a sample request.
9. Document your results and tradeoffs in the README.

## Repository layout

- `src/taxi_duration/`: data loading, cleaning rules, features, training, tuning and evaluation.
- `src/app/`: the FastAPI app (`/predict`, `/health`) and its request schema.
- `transform/`: dbt models that turn the raw trips into a clean table.
- `notebooks/`: the exploration notebook.
- `tests/`: unit, integration, model, behavioral and deploy tests.
- `docs/`: decisions (ADRs), the tuning plan, data sources and traceability.
- `reports/results/`: the lock, tuning trials and test-week metrics.
- `sample-request.json`: an example input for the API.
- Not in git: `data/` (the warehouse), `models/` (local model files), and `mlflow.db` and `mlartifacts/` (the MLflow store).

This project uses `src/app/`, and keeps models in MLflow. MLflow runs on port 5001, because macOS uses 5000 for AirPlay Receiver.

## Local-First Deployment Guidance

Choose **one** of these local deployment paths:

- Run the API directly with `uvicorn`.
- Package the API in Docker and run it locally.

Keep the deployment workflow easy to build, run and review on a single machine.

## Local Data Services

- `MLflow`: Run experiment tracking and inspect runs locally, for example at `http://127.0.0.1:5000`.
- `FastAPI`: Serve predictions locally, for example at `http://127.0.0.1:8000`.
- `Docker`: Optional, if you want a containerized local workflow.

## Questions

1. **What is the RMSE of your final model?**

   4.11 minutes on the test week (January 25 to 31, 791,334 trips), scored once. For comparison, the default random forest scored 4.16 minutes, and always predicting the average trip time scored 9.59 minutes. The final model is also about 15 times smaller than the default one (184 MB instead of 2.7 GB). See [Results](README.md#results) and [Caveats](README.md#caveats).

2. **What would you do differently if you had more time?**
   - Build the size limit into the search from the start. Before tuning, I set a 300 MB limit on the model file, so the API can load it quickly on a laptop. In this run, 23 of 30 trials produced models over that limit, so most of the tuning budget went to models I could not deploy. A new plan, written before any results, would search only among deployable sizes and be tested on a fresh month, because the January test week has already been used.
   - Try separate models for airport trips and long-distance trips, chosen by pickup zone, dropoff zone and distance, which are known at pickup. First check whether one model with extra features (an airport flag, distance bands) closes the gap, because each extra model must be trained, registered and monitored separately.
   - Package the API in Docker. I chose the simpler path the project allows (running it with `uvicorn`). Docker first needs a security decision: inside a container, `127.0.0.1` is the container itself, so it can't reach the local MLflow server.

   See [Follow-up ideas](README.md#follow-up-ideas) for the full list.

3. **Was tuning worth it?**

|                       | Baseline (default forest) | Tuned model | Change                         |
| --------------------- | ------------------------- | ----------- | ------------------------------ |
| Test-week RMSE        | 4.16 min                  | 4.11 min    | 1.2% better                    |
| Test-week MAE         | 2.80 min                  | 2.74 min    | 2% better                      |
| Cross-validation RMSE | 4.31 min                  | 4.24 min    | 1.6% better, on all 3 folds    |
| Model file size       | 2,694 MB                  | 184 MB      | about 15 times smaller         |
| One prediction        | about 2 ms                | about 3 ms  | both far under the 50 ms limit |

**For accuracy alone, no.** The tuned model is only slightly more accurate: about 3 seconds better per trip on average. That is below the 2% improvement I set in advance as "worth it". It is also slightly worse on long trips of 30 to 60 minutes (10.16 vs 9.88 minutes RMSE).

**For deployment, yes.** The default forest is 2.7 GB, far over the 300 MB limit I set before tuning, so it could not be served from a laptop API. The tuned model keeps the same accuracy at a fifteenth of the size, which is what made deployment practical.

**The cost was small:** 30 trials ran in under 15 minutes. Most of the extra work was in the rules around tuning (a plan written in advance, size and speed checks, a test week used only once), and that work makes the result trustworthy.

**Verdict:** tuning was worth it, but because it made the model deployable, not because it made it much more accurate.
