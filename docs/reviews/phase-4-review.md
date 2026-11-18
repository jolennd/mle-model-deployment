# Training step review (train and lock)

Verdict: FIX LIST (no block). Items 1 and 2 are small and should be done before the human approves the gate.

## Passed, with evidence
- Lock alone: `git --no-pager log --stat --oneline` shows 42d7701 touching only reports/results/baseline_lock.json (39 lines). It comes after 05ed8fd (green) and before 1e54ead. No commit reads test rows.
- No test-row reads: grep for `load_test` in .py, .mk, Makefile and .md outside .venv and the kit folder finds it only in src/taxi_duration/splits.py (definition, lines 3 and 39) and tests/unit/test_splits_gates.py. No evaluation code exists yet. `TEST_START` is used only in config.py and splits.py (load_train upper bound, load_test). `load_test` checks the lock path before opening the database (splits.py lines 45 to 49).
- One scope function: labels.apply_scope (labels.py line 49). load_train and load_test both call it (splits.py lines 35 and 51). Bounds are only in config.py lines 22 to 27.
- Row counts: I ran labels.scope_report on the training window read through splits._read. Output matches the ADR 7 table: input 2641485; duration 54065 added; distance 56282; speed 134 (alone 1290); zones 11135; embargo 1902; kept 2517967.
- Fitted steps: the OrdinalEncoder sits in the pipeline (features.py lines 33 to 52), and cross_validate fits a fresh pipeline per fold on the training fold only (train.py lines 62 to 68). Folds apply the embargo (`dropoff_at < window_start`, splits.py line 79).
- Red before green: 70c9727 -> 245ecc4; c85a618 -> b07dd73; da61b74 -> ef2298b; 993ad27 -> 05ed8fd; 86285fc -> 6b38675; 9accbc1 -> working tree. Batches hold 3 to 5 tests (test_ prefix count per red commit: 5, 3, 4, 5, 5, 3).
- Tests not weakened: `git --no-pager diff 70c9727 HEAD --stat -- tests` shows one changed existing line (tests/unit/test_labels.py, 1 line, the import order, 62b80fa). Everything else is added files.
- cases.toml: `git --no-pager diff 245ecc4..HEAD -- tests/behavioral/cases.toml` is empty (0 lines).
- Config: grep of src finds no secrets, hosts or ports. DB only via `require_env("DB_CONN")` in db.py line 28. `MLFLOW_TRACKING_URI` is read from the environment (train.py line 155). The loopback hosts appear in .env.example only.
- skops: `artifacts.load_model(folder, trusted_types)` passes an explicit list (artifacts.py line 45). The allow list is `MODEL_TRUSTED_TYPES=sklearn.tree._tree.Tree` (.env.example line 20). No "trust all", and the file's own list is not used.
- CLI (working tree): commands parse, call `training.run_baseline` or `training.read_lock`, and print. Results via click.echo, logs via logging (stderr), expected errors via ClickException. The committed HEAD version of `evaluate` still read and parsed the lock itself; the uncommitted change moves that into train.read_lock. It must be committed.
- ruff: `uv run ruff check .` passes; `ruff format --check .` reports 65 files already formatted.
- `make validate-gates`: 63 passed, 15 deselected, 1 warning (mlflow/sqlalchemy deprecation).
- Accepted by the human, not reopened: baseline not registered (2,694,430,930 bytes), behavioral tests deferred (ADR 8 point 10), API distance range 0.1 to 80, source hash record read by train, cv_folds.json git-ignored (`git check-ignore` confirms .gitignore line 195) and checked by hashes in the lock.
- Lock values (baseline_lock.json): mean_rmse 4.311476785887606, mean_reference_mean_rmse 9.77688704610776, sample_rows 300000, seed 42.

## Fix list
1. Registry verify files not copied (step 5 of the training step). `find` finds verify_model.py only in the kit's variants folder; none in src/taxi_duration or tests. ADR 8 point 3 excuses registration, not the verify files or their tests. Either copy them with tests, or record in an ADR (or ADR 8) that this moves to the tuning step with the registered model. Traceability R10 already lists the tests as open.
2. Two test files were not committed red first: tests/unit/test_labels_scope.py came in 62b80fa after the green 245ecc4, and tests/unit/test_scope_report.py came inside the green commit b07dd73. Both pass and cover the embargo, same-function and row-count requirements. Record this as a process deviation in the phase status file. The history cannot be changed.
3. Owner mixing in commits: 245ecc4 holds labels.py, config.py (shared), ADRs and cases.toml; b07dd73 holds features.py, splits.py and a test; 6b38675 holds cli.py, train.py, artifacts.py and pyproject.toml. The rule is one commit per owner. For the pending work, commit separately: (a) src/taxi_duration/train.py and cli.py (green for read_lock), (b) docs/01-traceability.md, (c) docs/reviews/phase-4-review.md.
4. `make check-kit` does not exist in Makefile or make/*.mk (searched both), so the tool-name check was not run by command. The manual check found tool names (mlflow, skops, click, duckdb, sklearn) in src only, which is allowed; the rule covers the kit's own phase and core files.
5. Minor, no action required before the gate: `n_estimators=100` is hard-coded in cli.py line 52 and train.py `run_baseline`, not in config.py. train.py lines 133 and 134 call the private helpers `source_hash._project_root` and `_read_record`. Consider a public helper in the tuning step.

## Files to commit (human)
- src/taxi_duration/train.py, src/taxi_duration/cli.py
- docs/01-traceability.md
- docs/reviews/phase-4-review.md
