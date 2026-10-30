"""CPU protocol checks only; synthetic objectives are not GPU evidence."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from bo_kernel_tuner.configs.search_spaces import Workload, grid_size, search_space
from bo_kernel_tuner.tuner.engine import EvaluationFailure, run_search
from bo_kernel_tuner.tuner.hardware import prune_reason
from runpod import launch


@pytest.mark.parametrize("transfer,studies,attempts", [(False, 12, 216), (True, 16, 288)])
def test_pilot_dry_run_without_cuda_or_writes(tmp_path, transfer, studies, attempts):
    output = tmp_path / "target"
    args = ["pilot", "--dry-run", "--output", str(output)]
    if transfer:
        args += ["--source-run", str(tmp_path / "future-source"), "--top-k", "3"]
    # Fail even on importing CUDA libraries: planning must work on a CPU-only machine.
    script = (
        "import sys; sys.modules['torch'] = None; sys.modules['triton'] = None; "
        "from runpod.launch import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["studies"] == studies and plan["max_attempts"] == attempts
    assert set(plan["max_attempts_by_method"].values()) == {72}
    assert plan["candidate_worker_timeout_sum_s"] == attempts * 60
    assert plan["output"] == str(output)
    assert plan["timing"] == str(tmp_path / "target.timing.json")
    assert len({p["output"] for p in plan["plan"]}) == studies
    assert all(Path(p["output"]).is_relative_to(output) for p in plan["plan"])
    assert not list(tmp_path.iterdir())


def test_pilot_protocol_and_legal_domains(hardware):
    from dataclasses import replace

    from bo_kernel_tuner.configs.search_spaces import grid_configs

    config = launch.load_config("pilot")
    assert config["seeds"] == [0, 1]
    assert config.get("time_budget") is None  # Avoid method-dependent time censoring.
    assert config["measurement"]["validation_seeds"] == 2
    assert {c["workload"]["kernel"] for c in config["cases"]} == {"gemm", "rmsnorm"}
    for case in config["cases"]:
        assert case["profile"] == "full" and case["workload"]["dtype"] == "fp16"
        assert 1 <= case["startup_trials"] < case["trials"]
        space = search_space(case["workload"]["kernel"], case["profile"])
        assert case["trials"] <= grid_size(space)
        # Static legality only. Does not assert compilation, correctness or performance.
        for capability in ((8, 6), (12, 0)):
            for params in grid_configs(space):
                assert (
                    prune_reason(
                        replace(hardware, capability=capability),
                        case["workload"]["kernel"],
                        params,
                        case["workload"]["n"],
                    )
                    is None
                )


@pytest.mark.parametrize("method", ["grid", "random", "tpe"])
def test_pilot_real_sampler_budgets_with_cpu_objective(tmp_path, hardware, method):
    config = launch.load_config("pilot")
    for case in config["cases"]:
        for seed in config["seeds"]:
            calls = []

            def synthetic(params):
                calls.append(params)
                if len(calls) == 1:
                    raise EvaluationFailure("deliberate CPU failure consumes one attempt")
                return {"latency_ms": float(sum(params.values()))}

            result = run_search(
                Workload(**case["workload"]),
                hardware,
                search_space(case["workload"]["kernel"], case["profile"]),
                method,
                seed,
                case["trials"],
                tmp_path / f"{case['id']}-{seed}",
                startup_trials=case["startup_trials"],
                evaluator=synthetic,
            )
            assert len(calls) == result["attempts"] == case["trials"]
            assert result["successful_evaluations"] == case["trials"] - 1
            assert Counter(t["state"] for t in result["trials"])["FAIL"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("trials", 0),
        ("trials", True),
        ("trials", 2.5),
        ("startup_trials", 25),
        ("profile", "typo"),
        ("m", 2.5),
        ("n", 0),
        ("warmup", True),
        ("repeats", 3.5),
        ("trial_timeout", None),
        ("trial_timeout", True),
        ("trial_timeout", float("inf")),
    ],
)
def test_invalid_pilot_values_rejected(tmp_path, field, value):
    config = launch.load_config("pilot")
    if field in ("m", "n"):
        config["cases"][0]["workload"][field] = value
    elif field in ("warmup", "repeats"):
        config["measurement"][field] = value
    elif field == "trial_timeout":
        config[field] = value
    else:
        config["cases"][0][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        launch.load_config(str(path))


@pytest.mark.parametrize(
    "kind",
    [
        "occupied",
        "file",
        "parent-file",
        "source",
        "source-child",
        "source-parent",
        "archive",
        "timing",
        "resume-missing",
        "top-k",
    ],
)
def test_dry_run_rejects_unsafe_paths_and_transfer_budget(tmp_path, capsys, kind):
    output = tmp_path / "output"
    args = ["pilot", "--dry-run", "--output", str(output)]
    if kind == "occupied":
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    elif kind == "file":
        output.write_text("keep")
    elif kind == "parent-file":
        output.write_text("keep")
        args[-1] = str(output / "child")
    elif kind.startswith("source"):
        source = {"source": output, "source-child": output / "child", "source-parent": tmp_path}[
            kind
        ]
        args += ["--source-run", str(source)]
    elif kind in ("archive", "timing"):
        (tmp_path / ("output.tar.gz" if kind == "archive" else "output.timing.json")).write_text(
            "keep"
        )
    elif kind == "resume-missing":
        args += ["--resume"]
    else:
        args += ["--source-run", str(tmp_path / "source"), "--top-k", "13"]
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert launch.main(args) == 1
    assert "failed" in capsys.readouterr().err
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_custom_pilot_path_matches_alias(tmp_path):
    path = tmp_path / "custom.json"
    config = launch.load_config("pilot")
    path.write_text(json.dumps(config))
    assert launch.load_config(str(path)) == config
