"""CPU protocol/failure tests. Synthetic fixtures are never GPU benchmark evidence."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from bo_kernel_tuner.configs.search_spaces import Workload, grid_configs, search_space
from bo_kernel_tuner.tuner import confirmation as c
from bo_kernel_tuner.tuner.worker import (
    BASELINES,
    MEASUREMENT_PROTOCOL,
    PRECISION_POLICY,
    Measurement,
    sample_statistics,
)


def timing(implementation="triton", repeats=3, value=1.0, dtype="fp16"):
    rtol, atol = (0.01, 0.02) if dtype == "fp16" else (0.02, 0.15)
    return {
        "status": "ok",
        "implementation": implementation,
        "protocol": MEASUREMENT_PROTOCOL,
        "precision_policy": PRECISION_POLICY,
        **sample_statistics([value] * repeats),
        "validation": [{"max_abs_error": 0.0, "rtol": rtol, "atol": atol}] * 2,
    }


@pytest.fixture
def source(tmp_path, hardware, monkeypatch):
    hardware = replace(hardware, uuid="CPU-TEST-DOUBLE")
    w = Workload("rmsnorm", 37, 77)
    space = search_space(w.kernel, "smoke")
    best = {
        "number": 0,
        "state": "COMPLETE",
        "value": 1.0,
        "params": next(grid_configs(space)),
        "attrs": {"measurement": timing()},
    }
    data = {
        "schema_version": 1,
        "status": "complete",
        "best": best,
        "trials": [best],
        "attempts": 1,
        "successful_evaluations": 1,
        "metadata": {
            "workload": w.signature(),
            "measurement": asdict(Measurement(1, 3, 1, 2)),
            "search_space": space,
            "code_fingerprint": c.code_fingerprint(),
            "protocol": "cold_compile_per_candidate;" + MEASUREMENT_PROTOCOL,
            "precision_policy": PRECISION_POLICY,
            "hardware": hardware.to_dict(),
            "environment": {"versions": {"test_double": "no-GPU-execution"}},
        },
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(c.Hardware, "detect", lambda _: hardware)
    monkeypatch.setattr(c, "environment", lambda: data["metadata"]["environment"])
    calls = []

    def worker(workload, hw, params, measurement, timeout, logs, number, *, order):
        calls.append((workload.data_seed, params, order))
        return {
            "status": "ok",
            "order": order,
            "data_seed": workload.data_seed,
            "hardware": hw.to_dict(),
            "candidate": timing(value=2.0),
            "torch": timing(BASELINES[workload.kernel], value=4.0),
        }

    monkeypatch.setattr(c, "isolated_evaluate", worker)
    return path, data, hardware, calls


def test_confirmation_freezes_selection_uses_heldout_seeds_and_balances_order(source, tmp_path):
    path, data, _, calls = source
    output = tmp_path / "confirm"
    report = c.confirm(path, output)
    assert report["status"] == "complete" and len(calls) == 6
    assert report["search_best_ms"] == 1.0
    assert report["summary"]["candidate_median_ms"] == 2.0  # Never substitute search best.
    assert report["summary"]["median_paired_speedup"] == 2.0
    assert [x[0] for x in calls] == [19, 21, 23, 25, 27, 29]
    assert all(x[1] == data["best"]["params"] for x in calls)
    assert [x[2][0] for x in calls] == ["candidate", "torch"] * 3
    assert (output / "source-result.json").read_bytes() == path.read_bytes()
    assert c.verify_confirmation(output) == report["summary"]


def test_dry_run_and_verify_need_no_cuda(source, tmp_path, monkeypatch):
    path, _, _, _ = source
    output = tmp_path / "confirm"
    c.confirm(path, output, repetitions=3)
    monkeypatch.setattr(c.Hardware, "detect", lambda _: pytest.fail("CPU path touched CUDA"))
    assert c.main(["--verify", str(output)]) == 0
    target = tmp_path / "planned"
    plan = c.confirm(path, target, dry_run=True)
    assert len(plan["plan"]) == 6 and not target.exists()


@pytest.mark.parametrize("mutation", ["fingerprint", "winner", "samples", "counts", "protocol"])
def test_invalid_source_rejected_before_device_access(source, tmp_path, monkeypatch, mutation):
    path, data, _, _ = source
    if mutation == "fingerprint":
        data["metadata"]["code_fingerprint"] = "old"
    elif mutation == "winner":
        data["best"] = dict(data["best"], number=99)
    elif mutation == "samples":
        data["best"]["attrs"]["measurement"]["samples_ms"] = [float("nan")] * 3
    elif mutation == "counts":
        data["attempts"] = 2
    else:
        data["metadata"]["protocol"] = "old-hot-buffer-protocol"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(c.Hardware, "detect", lambda _: pytest.fail("Invalid source reached CUDA"))
    with pytest.raises(ValueError):
        c.confirm(path, tmp_path / "confirm")


@pytest.mark.parametrize("failure", [RuntimeError("worker_timeout"), KeyboardInterrupt()])
def test_failure_preserves_partial_pairs_without_speedup(source, tmp_path, monkeypatch, failure):
    path, _, _, _ = source
    real = c.isolated_evaluate

    def worker(*args, **kwargs):
        if args[6] == 1:
            raise failure
        return real(*args, **kwargs)

    monkeypatch.setattr(c, "isolated_evaluate", worker)
    output = tmp_path / "confirm"
    with pytest.raises(type(failure)):
        c.confirm(path, output)
    report = json.loads((output / "confirmation.json").read_text())
    assert report["status"] in ("failed", "interrupted")
    assert len(report["pairs"]) == 1 and report["summary"] is None
    assert not (output / "receipt.json").exists()
    with pytest.raises(ValueError, match="completeness"):
        c.verify_confirmation(output)


@pytest.mark.parametrize(
    "mutation", ["summary", "seed", "order", "hardware", "median", "missing_pair", "tolerance"]
)
def test_semantic_tampering_is_rejected(source, tmp_path, mutation):
    path, _, _, _ = source
    output = tmp_path / "confirm"
    report = c.confirm(path, output)
    if mutation == "summary":
        report["summary"]["median_paired_speedup"] = 100.0
    elif mutation == "seed":
        report["pairs"][0]["data_seed"] = 17
    elif mutation == "order":
        report["pairs"][0]["order"].reverse()
    elif mutation == "hardware":
        report["pairs"][0]["hardware"]["uuid"] = "other-device"
    elif mutation == "median":
        report["pairs"][0]["candidate"]["latency_ms"] = 0.01
    elif mutation == "missing_pair":
        report["pairs"].pop()
    else:
        report["pairs"][0]["candidate"]["validation"][0]["atol"] = 100
    (output / "confirmation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        c.verify_confirmation(output)


def test_other_device_or_versions_and_overwrite_are_rejected(source, tmp_path, monkeypatch):
    path, _, hardware, _ = source
    output = tmp_path / "confirm"
    c.confirm(path, output)
    before = (output / "confirmation.json").read_bytes()
    with pytest.raises(ValueError, match="new confirmation"):
        c.confirm(path, output)
    assert (output / "confirmation.json").read_bytes() == before
    monkeypatch.setattr(c.Hardware, "detect", lambda _: replace(hardware, uuid="different"))
    with pytest.raises(ValueError, match="identity"):
        c.confirm(path, tmp_path / "wrong-device")
    monkeypatch.setattr(c.Hardware, "detect", lambda _: hardware)
    monkeypatch.setattr(c, "environment", lambda: {"versions": {"torch": "changed"}})
    with pytest.raises(ValueError, match="versions"):
        c.confirm(path, tmp_path / "wrong-version")


@pytest.mark.parametrize(
    "samples", [[1, 2], [1, 0, 2], [1, -1, 2], [1, float("inf"), 2], [1, float("nan"), 2]]
)
def test_invalid_cuda_samples_cannot_be_summarized(samples):
    with pytest.raises(ValueError):
        sample_statistics(samples)


@pytest.mark.parametrize("count", [True, 1.5, float("nan")])
def test_measurement_counts_are_strict_integers(count):
    with pytest.raises(ValueError, match="integers"):
        Measurement(warmup=count)


def test_bootstrap_uses_pairs_not_inner_cuda_samples():
    pairs = [{"candidate": timing(value=1), "torch": timing(value=v)} for v in (0.5, 1, 2)]
    result = c.summarize(pairs)
    more_samples = copy.deepcopy(pairs)
    for p in more_samples:
        p["candidate"]["samples_ms"] *= 100
        p["torch"]["samples_ms"] *= 100
    assert c.summarize(more_samples) == result
    assert result["paired_repetitions"] == 3
    assert result["ci95_low"] <= 1 <= result["ci95_high"]


@pytest.mark.parametrize("kernel", ["gemm", "rmsnorm", "moe_grouped_gemm"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (37, 77, 93)])
def test_torch_baseline_numerics_on_cpu(kernel, dtype, shape):
    torch = pytest.importorskip("torch")
    from bo_kernel_tuner.tuner.worker import check_output, prepare

    w = Workload(kernel, *shape, dtype, (0, 1, 7, 0, 3))
    call, _, ref = prepare(w, torch.device("cpu"), 19, 1, "torch")
    check_output(call({}), ref, dtype)
