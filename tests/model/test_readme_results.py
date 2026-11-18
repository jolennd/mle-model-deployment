"""README results section matches the committed evaluation files (R1, R6, R9).

Every expected number is computed from metrics.json and trials.json at test time.
"""

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parents[2]
SIZE_LIMIT = 300_000_000
SEGMENTS = ["duration_1_10", "duration_10_30", "duration_30_60", "airport", "other"]


@pytest.fixture(scope="module")
def metrics():
    return json.loads((ROOT / "reports/results/metrics.json").read_text())


@pytest.fixture(scope="module")
def trials():
    return json.loads((ROOT / "reports/results/trials.json").read_text())


@pytest.fixture(scope="module")
def readme():
    return (ROOT / "README.md").read_text()


def _section(text, prefix):
    """Text from the heading that starts with prefix to the next level 2 heading."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(prefix)), None)
    assert start is not None, f"README has no section '{prefix}'"
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start:end])


def _paragraphs(text):
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _table(section, header):
    """Rows (list of cell lists) of the table whose header cells equal header."""
    lines = [ln.strip() for ln in section.splitlines()]

    def cells(ln):
        return [c.strip() for c in ln.strip("|").split("|")]

    idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("|") and cells(ln) == header),
        None,
    )
    assert idx is not None, f"no table with header {header}"
    block = []
    for ln in lines[idx + 1 :]:
        if not ln.startswith("|"):
            break
        block.append(cells(ln))
    body = [r for r in block if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    return {r[0]: r for r in body}


def _has_paragraph(section, *needles):
    low = [n.lower() for n in needles]
    return any(all(n in p.lower() for n in low) for p in _paragraphs(section))


def _mean(xs):
    return sum(xs) / len(xs)


def test_readme_rmse_matches_evaluation_report(readme, metrics):
    results = _section(readme, "## Results")
    keys = {
        "Champion": "chosen",
        "Baseline": "baseline",
        "Mean reference": "mean_reference",
    }
    top = _table(results, ["Model", "RMSE", "MAE", "R2"])
    for label, key in keys.items():
        m = metrics[key]
        want = [label, f"{m['rmse']:.3f}", f"{m['mae']:.3f}", f"{m['r2']:.3f}"]
        assert top.get(label) == want
    seg_header = [
        "Model",
        "1 to 10 min",
        "10 to 30 min",
        "30 to 60 min",
        "Airport",
        "Other",
    ]
    seg = _table(results, seg_header)
    for label, key in keys.items():
        by = metrics[key]["rmse_by_segment"]
        want = [label] + [f"{by[s]:.3f}" for s in SEGMENTS]
        assert seg.get(label) == want
    assert f"{metrics['chosen']['mean_error']:.3f}" in results
    assert f"{metrics['rows_used']:,}" in results


def test_readme_states_distance_caveat(readme):
    text = _section(readme, "## Results") + "\n\n" + _section(readme, "## Caveats")
    assert _has_paragraph(text, "ADR 2", "optimistic", "distance")
    assert _has_paragraph(text, "ADR 7", "speed", "80 mph")


def test_readme_reports_both_test_scores_size_speed_and_promotion(
    readme, metrics, trials
):
    results = _section(readme, "## Results")
    final = trials["final"]
    info = metrics["model_info"]
    assert f"{final['size_bytes']:,}" in results
    assert f"{final['size_bytes'] / 1e6:.1f} MB" in results
    assert f"{final['predict_ms']:.1f} ms" in results
    assert _has_paragraph(
        results,
        f"version {info['version']}",
        info["alias"],
        info["sha256"],
        "2026-11-18",
    )
    for key, value in final["params"].items():
        assert str(key) in results
        assert str(value) in results


def test_readme_caveats_state_the_numbers_from_the_files(readme, metrics, trials):
    cav = _section(readme, "## Caveats")
    low = cav.lower()
    champ, base = metrics["chosen"], metrics["baseline"]
    final = trials["final"]
    by_name = {t["name"]: t for t in trials["trials"]}
    champ_cv = _mean(by_name[final["name"]]["fold_rmse"])
    base_cv = _mean([f["rmse"] for f in metrics["cv"]["baseline"]])
    test_gain = (base["rmse"] - champ["rmse"]) / base["rmse"] * 100
    cv_gain = (base_cv - champ_cv) / base_cv * 100
    assert _has_paragraph(cav, f"{test_gain:.1f}%", f"{cv_gain:.1f}%", "2%")
    assert "size limit" in low
    assert "300 MB" in cav
    seg_c = champ["rmse_by_segment"]
    seg_b = base["rmse_by_segment"]
    assert _has_paragraph(
        cav, f"{seg_c['duration_30_60']:.2f}", f"{seg_b['duration_30_60']:.2f}"
    )
    assert f"{champ['mean_error']:.2f}" in cav
    assert _has_paragraph(cav, f"{seg_c['airport']:.2f}", "airport")
    for phrase in ["one month", "300,000", "congestion pricing", "python version"]:
        assert phrase in low
    failing = [
        _mean(t["fold_rmse"])
        for t in trials["trials"]
        if t["size_bytes"] >= SIZE_LIMIT or t["predict_ms"] is None
    ]
    if failing and min(failing) < champ_cv:
        assert "over the size limit" in low


def test_champion_beats_the_mean_reference(metrics):
    """Characterization (green now) of the committed results, not a new behavior."""
    chosen, ref = metrics["chosen"], metrics["mean_reference"]
    assert chosen["rmse"] < ref["rmse"]
    assert chosen["r2"] > 0.5
    folds = zip(metrics["cv"]["chosen"], metrics["cv"]["mean_reference"], strict=True)
    for c, r in folds:
        assert c["rmse"] < r["rmse"]


def test_env_example_has_model_placeholders_without_a_hash():
    text = (ROOT / ".env.example").read_text()
    assert re.search(
        r"^MODEL_URI=models:/taxi-duration@champion\s*$", text, re.MULTILINE
    )
    assert re.search(r"^MODEL_SHA256=\s*$", text, re.MULTILINE)
    assert not re.search(r"[0-9a-fA-F]{64}", text)
