import json

import optuna
import pytest

from bo_kernel_tuner.configs.search_spaces import Workload, search_space
from bo_kernel_tuner.tuner.engine import EvaluationFailure, run_search, sampler_for
from bo_kernel_tuner.tuner.hardware import Hardware
from bo_kernel_tuner.tuner.transfer import load_warm_start


def objective(p):
    return {"latency_ms": float(p["ROWS_PER_CTA"] + p["num_warps"])}


def search(tmp_path, hardware, method="tpe", budget=4, evaluator=objective, **kwargs):
    return run_search(
        Workload("rmsnorm"),
        hardware,
        search_space("rmsnorm", "smoke"),
        method,
        0,
        budget,
        tmp_path,
        evaluator=evaluator,
        startup_trials=2,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["grid", "random", "tpe"])
def test_attempt_accounting_and_persistence(tmp_path, hardware, method):
    result = search(tmp_path, hardware, method)
    assert result["attempts"] == result["successful_evaluations"] == 4
    assert result["best"]["value"] >= 5
    assert json.loads((tmp_path / "result.json").read_text()) == result
    study = optuna.load_study(
        study_name="kernel", storage=f"sqlite:///{tmp_path / 'study.sqlite3'}"
    )
    assert len(study.trials) == 4
    assert all(t["attrs"]["elapsed_s"] > 0 for t in result["trials"])


def test_grid_exhaustion(tmp_path, hardware):
    result = search(tmp_path, hardware, "grid", 100)
    assert result["attempts"] == result["unique_successful_configs"] == 4


@pytest.mark.parametrize("method", ["random", "tpe"])
def test_all_samplers_share_grid_sized_budget_cap(tmp_path, hardware, method):
    result = search(tmp_path, hardware, method, 100)
    assert result["attempts"] == result["metadata"]["budget"] == 4
    assert result["metadata"]["requested_budget"] == 100


def test_no_silent_failures(tmp_path, hardware):
    def fail(p):
        raise EvaluationFailure("simulated timeout")

    result = search(tmp_path, hardware, evaluator=fail)
    assert result["status"] == "failed" and result["best"] is None
    assert all(t["state"] == "FAIL" and "failure" in t["attrs"] for t in result["trials"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_reject_nonpositive_or_nonfinite_measurement(tmp_path, hardware, value):
    result = search(tmp_path, hardware, budget=1, evaluator=lambda p: {"latency_ms": value})
    assert result["status"] == "failed"


def test_pruning_counts_toward_budget(tmp_path, hardware):
    space = {"ROWS_PER_CTA": [1], "num_warps": [64]}
    result = run_search(
        Workload("rmsnorm"),
        hardware,
        space,
        "grid",
        1,
        5,
        tmp_path,
        evaluator=lambda p: pytest.fail("Pruned config must not execute"),
    )
    assert result["attempts"] == 1
    assert result["trials"][0]["state"] == "PRUNED"


def test_unexpected_bug_aborts_with_checkpoint(tmp_path, hardware):
    def fail(p):
        raise KeyError("programming bug")

    with pytest.raises(KeyError):
        search(tmp_path, hardware, evaluator=fail)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "interrupted"


def test_refuse_overwrite(tmp_path, hardware):
    (tmp_path / "important.txt").write_text("retain")
    with pytest.raises(ValueError, match="empty"):
        search(tmp_path, hardware)
    assert (tmp_path / "important.txt").read_text() == "retain"


def test_warm_start_remeasures_and_consumes_budget(tmp_path, hardware):
    search(tmp_path / "source", hardware, "grid")
    w, space = Workload("rmsnorm"), search_space("rmsnorm", "smoke")
    target = Hardware("different GPU", (12, 0), 170, 32 * 1024**3, 99 * 1024)
    queue, provenance = load_warm_start(
        tmp_path / "source/result.json", w, space, 2, target, lambda p: None
    )
    calls = []

    def target_objective(p):
        calls.append(p)
        return {"latency_ms": 1000.0}

    result = search(
        tmp_path / "target",
        target,
        evaluator=target_objective,
        warm_params=queue,
        transfer=provenance,
    )
    assert calls[:2] == queue and len(calls) == 4
    assert result["best"]["value"] == 1000.0  # Source objective values never leak.
    assert result["trials"][0]["attrs"]["warm_start_rank"] == 0
    assert provenance["source_hardware"]["name"] != provenance["target_hardware"]["name"]


def test_transfer_compatibility_and_pruning(tmp_path, hardware):
    search(tmp_path / "source", hardware, "grid")
    path = tmp_path / "source/result.json"
    space = search_space("rmsnorm", "smoke")
    with pytest.raises(ValueError, match="mismatch"):
        load_warm_start(path, Workload("rmsnorm", n=7), space, 2, hardware, lambda p: None)
    with pytest.raises(ValueError, match="No legal"):
        load_warm_start(path, Workload("rmsnorm"), space, 2, hardware, lambda p: "illegal")
    data = json.loads(path.read_text())
    data["metadata"]["code_fingerprint"] = "stale"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="code differ"):
        load_warm_start(path, Workload("rmsnorm"), space, 2, hardware, lambda p: None)


def test_tpe_is_multivariate():
    sampler = sampler_for("tpe", {}, 1)
    assert sampler._multivariate is True


def test_isolated_timeout_and_crash(tmp_path, hardware, monkeypatch):
    import subprocess

    from bo_kernel_tuner.tuner.engine import isolated_evaluate
    from bo_kernel_tuner.tuner.worker import Measurement

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(EvaluationFailure, match="timeout"):
        isolated_evaluate(Workload(), hardware, {}, Measurement(), 1, tmp_path, 0)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], -11))
    with pytest.raises(EvaluationFailure, match="exit_-11"):
        isolated_evaluate(Workload(), hardware, {}, Measurement(), 1, tmp_path, 1)


def test_wall_budget_stops_after_attempt(tmp_path, hardware, monkeypatch):
    import time

    def slow(p):
        time.sleep(0.04)
        return objective(p)

    # Startup overhead is intentionally part of the budget, so zero attempts is valid.
    result = search(tmp_path, hardware, budget=100, evaluator=slow, time_budget=0.2)
    assert result["attempts"] < 100


def test_cli_dry_run_and_invalid_options(tmp_path):
    from typer.testing import CliRunner

    from bo_kernel_tuner.run_benchmark import app

    runner = CliRunner()
    result = runner.invoke(
        app, ["--dry-run", "--kernel", "moe_grouped_gemm", "--output", str(tmp_path / "unused")]
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "unused").exists()
    result = runner.invoke(app, ["--dry-run", "--seeds", "0,0"])
    assert result.exit_code == 2
