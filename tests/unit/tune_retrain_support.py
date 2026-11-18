"""Control what happens to saves, loads and timings in tune.MODEL_DIR only.

Trials save into a temporary folder and stay untouched. outcomes maps
(max_depth, n_estimators) to a dict with any of: "disk" (True: save_model raises a
disk RuntimeError), "size" (bytes returned by save_model), "ms" (predict time),
"load_error" and "ms_error" (exceptions raised by artifacts.load_model and by
tune.measure_predict_ms). Call it after setup_run.
"""

SMALL_SIZE = 1_000_000


def key(p):
    return (p["max_depth"], p["n_estimators"])


def control_model_dir(monkeypatch, outcomes):
    from taxi_duration import artifacts, tune

    real_save, real_load = artifacts.save_model, artifacts.load_model
    real_ms = tune.measure_predict_ms  # the stub from setup_run

    def save(fitted, folder):
        out = outcomes.get(key(fitted.params), {})
        if str(folder) != tune.MODEL_DIR or not out:
            return real_save(fitted, folder)
        if out.get("disk"):
            raise RuntimeError("not enough disk space to save the model")
        real_save(fitted, folder)
        return out.get("size", SMALL_SIZE)

    def load(folder, trusted):
        model = real_load(folder, trusted)
        if str(folder) == tune.MODEL_DIR:
            model.in_model_dir = True
            error = outcomes.get(key(model.params), {}).get("load_error")
            if error:
                raise error
        return model

    def measure(fitted, row):
        if getattr(fitted, "in_model_dir", False):
            out = outcomes.get(key(fitted.params), {})
            if out.get("ms_error"):
                raise out["ms_error"]
            if "ms" in out:
                return out["ms"]
        return real_ms(fitted, row)

    monkeypatch.setattr(artifacts, "save_model", save)
    monkeypatch.setattr(artifacts, "load_model", load)
    monkeypatch.setattr(tune, "measure_predict_ms", measure)
