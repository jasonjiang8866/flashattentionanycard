import argparse
import copy
import json
import tarfile
from pathlib import Path

import pytest

from bo_kernel_tuner.tuner import engine
from runpod import launch


def junit(path, child="", count=19):
    path.write_text(
        "<testsuites><testsuite>"
        + "".join(
            f'<testcase name="test_{i}">{child if i == 0 else ""}</testcase>' for i in range(count)
        )
        + "</testsuite></testsuites>"
    )


@pytest.mark.parametrize(
    "child,count", [("<skipped/>", 19), ("<failure/>", 19), ("<error/>", 19), ("", 0), ("", 18)]
)
def test_gpu_junit_cannot_pass_on_skips_or_incomplete_execution(tmp_path, child, count):
    path = tmp_path / "gpu-tests.xml"
    junit(path, child, count)
    with pytest.raises(RuntimeError):
        launch.validate_junit(path)


def test_gpu_junit_complete(tmp_path):
    path = tmp_path / "gpu-tests.xml"
    junit(path)
    assert launch.validate_junit(path) == 19


def test_suite_plans_and_cpu_dry_run(capsys, monkeypatch):
    monkeypatch.setattr(
        launch, "gpu_preflight", lambda _: pytest.fail("dry run must not touch CUDA")
    )
    assert launch.main(["smoke", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["studies"] == 18 and plan["max_attempts"] == 36
    assert launch.main(["performance", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["studies"] == 90 and plan["max_attempts"] == 3360


@pytest.mark.parametrize("mutation", ["path", "duplicate", "seed", "unknown", "nan"])
def test_bad_suite_config_rejected(tmp_path, mutation):
    cfg = launch.load_config("smoke")
    if mutation == "path":
        cfg["cases"][0]["id"] = "../escape"
    elif mutation == "duplicate":
        cfg["cases"][1]["id"] = cfg["cases"][0]["id"]
    elif mutation == "seed":
        cfg["seeds"] = [0, 0]
    elif mutation == "unknown":
        cfg["typo"] = True
    else:
        cfg["time_budget"] = float("nan")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        launch.load_config(str(path))


@pytest.fixture
def suite(tmp_path, hardware, monkeypatch):
    # Deliberate CPU test doubles, never exported as user GPU benchmark results.
    config = launch.load_config("smoke")
    config["cases"] = [next(c for c in config["cases"] if c["id"] == "rmsnorm_fp16")]
    config["cases"][0]["trials"] = 1
    config["measurement"].update(repeats=3, warmup=1, batch=1)
    args = argparse.Namespace(device=0, source_run=None, source_seed=0, top_k=1, resume=False)
    monkeypatch.setattr(launch, "gpu_preflight", lambda _: hardware)
    monkeypatch.setattr(launch, "run_logged", lambda command, path: path.write_text("fixture\n"))

    def fake_gpu_tests(output, device):
        junit(output / "gpu-tests.xml")
        (output / "gpu-tests.log").write_text("CPU TEST DOUBLE\n")
        return launch.validate_junit(output / "gpu-tests.xml")

    monkeypatch.setattr(launch, "gpu_tests", fake_gpu_tests)
    real_search = engine.run_search
    calls = []

    def search(*pos, **kwargs):
        calls.append(pos[6])
        kwargs["evaluator"] = lambda p: {
            "status": "ok",
            "latency_ms": 1.0,
            "samples_ms": [1.0] * 3,
            "validation": [{"max_abs_error": 0.0}] * 2,
        }
        return real_search(*pos, **kwargs)

    monkeypatch.setattr(engine, "run_search", search)
    return args, config, tmp_path / "run", calls, search


def test_full_suite_artifacts_and_verified_resume(suite):
    args, config, output, calls, _ = suite
    launch.execute(args, config, output)
    status = json.loads((output / "status.json").read_text())
    assert status["gpu_validation_passed"] and status["completed_studies"] == 3
    assert status["gpu_test_count"] == 19
    with tarfile.open(output.with_suffix(".tar.gz")) as tar:
        assert "run/checksums.json" in tar.getnames()
    for path, digest in json.loads((output / "checksums.json").read_text()).items():
        assert launch.sha256(output / path) == digest
    assert len(calls) == 3
    args.resume = True
    launch.execute(args, config, output)
    assert len(calls) == 3


def test_missing_artifact_blocks_successful_resume(suite):
    args, config, output, _, _ = suite
    launch.execute(args, config, output)
    next(output.rglob("trials.csv")).unlink()
    args.resume = True
    with pytest.raises(RuntimeError, match="Incomplete"):
        launch.execute(args, config, output)
    assert not json.loads((output / "status.json").read_text())["gpu_validation_passed"]


def test_resume_mismatch_does_not_modify_existing_run(suite):
    args, config, output, _, _ = suite
    launch.execute(args, config, output)
    before = (output / "status.json").read_bytes()
    args.resume = True
    changed = copy.deepcopy(config)
    changed["measurement"]["batch"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        launch.execute(args, changed, output)
    assert (output / "status.json").read_bytes() == before


def test_interrupted_study_is_archived_and_restarted(suite, monkeypatch):
    args, config, output, calls, search = suite
    count = 0

    def interrupted(*pos, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            directory = Path(pos[6])
            directory.mkdir(parents=True)
            (directory / "result.json").write_text('{"status":"running"}')
            raise KeyboardInterrupt()
        return search(*pos, **kwargs)

    monkeypatch.setattr(engine, "run_search", interrupted)
    with pytest.raises(KeyboardInterrupt):
        launch.execute(args, config, output)
    assert json.loads((output / "status.json").read_text())["status"] == "interrupted"
    monkeypatch.setattr(engine, "run_search", search)
    args.resume = True
    launch.execute(args, config, output)
    assert len(calls) == 3
    assert len(list((output / "incomplete").rglob("result.json"))) == 1


def test_gpu_failure_preserves_diagnostics_without_success(suite, monkeypatch):
    args, config, output, calls, _ = suite

    def fail(output, device):
        junit(output / "gpu-tests.xml", "<skipped/>")
        return launch.validate_junit(output / "gpu-tests.xml")

    monkeypatch.setattr(launch, "gpu_tests", fail)
    with pytest.raises(RuntimeError, match="fully pass"):
        launch.execute(args, config, output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and not status["gpu_validation_passed"]
    assert not calls and output.with_suffix(".tar.gz").exists()


def test_worker_measurement_evidence_is_required(suite):
    args, config, output, _, _ = suite
    launch.execute(args, config, output)
    item = launch.make_plan(config)[0]
    directory = output / item["path"]
    path = directory / "result.json"
    result = json.loads(path.read_text())
    result["trials"][0]["attrs"]["measurement"]["samples_ms"] = []
    path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="measurement evidence"):
        launch.verify_study(directory, result["metadata"], require_receipt=False)


def test_suite_transfer_remeasures_source_candidates(suite, monkeypatch):
    from dataclasses import replace

    args, config, output, calls, _ = suite
    launch.execute(args, config, output)
    source_hardware = launch.gpu_preflight(0)
    monkeypatch.setattr(
        launch, "gpu_preflight", lambda _: replace(source_hardware, name="target CPU test double")
    )
    args.source_run = output
    target = output.parent / "target"
    launch.execute(args, config, target)
    assert len(calls) == 7
    result = json.loads(next(target.rglob("tpe_warm-seed-0/result.json")).read_text())
    assert (
        result["metadata"]["transfer"]["source_hardware"]["name"]
        != result["metadata"]["hardware"]["name"]
    )
    assert result["trials"][0]["attrs"]["warm_start_rank"] == 0


def test_gpu_checks_remap_selected_device_uuid(tmp_path, hardware, monkeypatch):
    from dataclasses import replace

    from bo_kernel_tuner.tuner.hardware import Hardware

    selected = replace(hardware, device=2, uuid="GPU-example-uuid")
    monkeypatch.setattr(
        Hardware, "detect", lambda device: selected if device == 2 else pytest.fail("wrong device")
    )

    def command(args, path, env):
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-example-uuid"
        junit(tmp_path / "gpu-tests.xml")

    monkeypatch.setattr(launch, "run_logged", command)
    assert launch.gpu_tests(tmp_path, 2) == 19
