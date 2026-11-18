"""Restart message (review 4, item 1).

Interface tune.run_tuning must provide: when trials.json already exists it raises
FileExistsError, before any MLflow call, whose message contains
"delete <path of trials.json> to restart; earlier MLflow runs remain".
With the default results folder this is exactly
"delete reports/results/trials.json to restart; earlier MLflow runs remain".
"""

import json
from pathlib import Path

import pytest
from tune_run_support import params, setup_run

SENTENCE = "to restart; earlier MLflow runs remain"


def test_default_folder_message(monkeypatch, tmp_path):
    from taxi_duration import config, tune

    seen = setup_run(monkeypatch, tmp_path, trials=[params(depth=12)])
    monkeypatch.setattr(config, "RESULTS_DIR", "reports/results")
    monkeypatch.chdir(tmp_path)
    old = Path("reports/results/trials.json")
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"keep": "me"}))

    with pytest.raises(FileExistsError) as caught:
        tune.run_tuning()

    expected = (
        "delete reports/results/trials.json to restart; earlier MLflow runs remain"
    )
    assert expected in str(caught.value)
    assert seen.client.runs == []
    assert seen.experiment_calls == []


def test_other_folder_message_names_that_path(monkeypatch, tmp_path):
    from taxi_duration import tune

    seen = setup_run(monkeypatch, tmp_path, trials=[params(depth=12)])
    seen.results.mkdir(parents=True)
    path = seen.results / "trials.json"
    path.write_text(json.dumps({"keep": "me"}))

    with pytest.raises(FileExistsError) as caught:
        tune.run_tuning()

    assert f"delete {path} {SENTENCE}" in str(caught.value)
