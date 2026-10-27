import numpy as np
import pytest

from bo_kernel_tuner.analysis.plot_metrics import generate, step_values, summarize
from bo_kernel_tuner.tests.test_engine import search


def test_step_curve_has_no_future_leakage():
    values = step_values(np.array([2, 4]), np.array([10, 5]), np.array([1, 2, 3, 4]))
    assert np.isinf(values[0])
    assert list(values[1:]) == [10, 10, 5]


def test_figures_and_censored_targets(tmp_path, hardware):
    a = search(tmp_path / "grid", hardware, "grid")
    b = search(tmp_path / "random", hardware, "random", evaluator=lambda p: {"latency_ms": 100})
    summary = summarize([a, b])
    assert not summary.iloc[1].target_reached
    assert np.isnan(summary.iloc[1].seconds_to_target)
    assert generate([tmp_path / "grid", tmp_path / "random"], tmp_path / "figures") == 1
    assert len(list((tmp_path / "figures").rglob("*.pdf"))) == 3


def test_hardware_cohorts_are_separate(tmp_path, hardware):
    from dataclasses import replace

    search(tmp_path / "a", hardware, "grid")
    search(tmp_path / "b", replace(hardware, name="other", capability=(12, 0)), "grid")
    assert generate([tmp_path / "a", tmp_path / "b"], tmp_path / "figures") == 2


def test_duplicate_seed_rejected(tmp_path, hardware):
    search(tmp_path / "a", hardware, "grid")
    search(tmp_path / "b", hardware, "grid")
    with pytest.raises(ValueError, match="Duplicate"):
        generate([tmp_path / "a", tmp_path / "b"], tmp_path / "figures")


def test_paired_ratios_do_not_drop_censored_seeds():
    import pandas as pd

    from bo_kernel_tuner.analysis.plot_metrics import paired_comparisons

    rows = []
    for seed in range(3):
        for method, value in [("grid", 10.0), ("tpe", 5.0)]:
            rows.append(
                dict(
                    method=method,
                    seed=seed,
                    best_ms=value,
                    attempts_to_target=value,
                    seconds_to_target=value,
                    target_reached=True,
                )
            )
    frame = pd.DataFrame(rows)
    result = paired_comparisons(frame)
    assert (result.median_ratio == 2).all()
    assert (result.ci95_low == 2).all()
    frame.loc[(frame.method == "tpe") & (frame.seed == 2), "seconds_to_target"] = np.nan
    frame.loc[(frame.method == "tpe") & (frame.seed == 2), "target_reached"] = False
    row = paired_comparisons(frame).query("metric == 'seconds_to_target'").iloc[0]
    assert np.isnan(row.median_ratio)
    assert row.observed_pairs == 2
    assert row.candidate_hit_rate == pytest.approx(2 / 3)
