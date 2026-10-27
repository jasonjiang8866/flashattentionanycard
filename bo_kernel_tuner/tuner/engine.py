"""Sequential, budget-matched search with complete attempt accounting."""

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import optuna

from bo_kernel_tuner.configs.search_spaces import grid_size, validate_params
from bo_kernel_tuner.tuner.hardware import prune_reason, resource_estimate
from bo_kernel_tuner.tuner.worker import Measurement


class EvaluationFailure(RuntimeError):
    pass


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for sub in ("kernels", "configs", "tuner"):
        for path in sorted((root / sub).glob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    versions = {}
    for name in ("torch", "triton", "optuna", "pandas", "matplotlib", "seaborn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.limit",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        smi = "unavailable"
    return {
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi_snapshot": smi,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "triton_environment": {k: v for k, v in os.environ.items() if k.startswith("TRITON_")},
    }


def sampler_for(method, space, seed, startup_trials=10):
    if method == "grid":
        return optuna.samplers.GridSampler(space, seed=seed)
    if method == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if method == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed, multivariate=True, n_startup_trials=startup_trials, constant_liar=False
        )
    raise ValueError("method must be grid, random, or tpe")


def isolated_evaluate(workload, hardware, params, measurement, timeout, log_dir, number):
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bo-kernel-") as directory:
        root = Path(directory)
        source, destination = root / "input.json", root / "output.json"
        source.write_text(
            json.dumps(
                {
                    "workload": asdict(workload),
                    "device": hardware.device,
                    "params": params,
                    "measurement": asdict(measurement),
                }
            )
        )
        env = dict(os.environ)
        # Every evaluated candidate starts with an empty disk JIT cache, including repeats.
        env["TRITON_CACHE_DIR"] = str(root / "cache")
        env["CUDA_CACHE_PATH"] = str(root / "cuda-cache")
        if env.get("TRITON_INTERPRET", "0") != "0":
            raise EvaluationFailure("TRITON_INTERPRET must be disabled for GPU measurement")
        with (log_dir / f"trial-{number}.log").open("w") as log:
            try:
                run = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "bo_kernel_tuner.tuner.worker",
                        str(source),
                        str(destination),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise EvaluationFailure(f"worker_timeout_after_{timeout}s") from exc
        if run.returncode or not destination.exists():
            raise EvaluationFailure(f"worker_exit_{run.returncode}; see logs/trial-{number}.log")
        result = json.loads(destination.read_text())
        if result["status"] != "ok":
            raise EvaluationFailure(f"{result['error_type']}: {result['error']}")
        return result


def _atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def export(study, output, metadata, status, elapsed):
    trials = [
        {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "attrs": t.user_attrs,
        }
        for t in study.trials
        if t.state != optuna.trial.TrialState.WAITING
    ]
    completed = [t for t in trials if t["state"] == "COMPLETE"]
    best = min(completed, key=lambda t: t["value"]) if completed else None
    data = {
        "schema_version": 1,
        "status": status,
        "metadata": metadata,
        "elapsed_s": elapsed,
        "attempts": len(trials),
        "successful_evaluations": len(completed),
        "unique_successful_configs": len(
            {json.dumps(t["params"], sort_keys=True) for t in completed}
        ),
        "best": best,
        "trials": trials,
    }
    _atomic_json(output / "result.json", data)
    with (output / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["number", "state", "latency_ms", "params", "attrs"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "number": t["number"],
                "state": t["state"],
                "latency_ms": t["value"],
                "params": json.dumps(t["params"]),
                "attrs": json.dumps(t["attrs"]),
            }
            for t in trials
        )
    return data


def run_search(
    workload,
    hardware,
    space,
    method,
    seed,
    budget,
    output,
    measurement=None,
    time_budget=None,
    trial_timeout=180,
    startup_trials=10,
    warm_params=(),
    transfer=None,
    heuristic_smem=False,
    evaluator=None,
):
    if budget < 1 or seed < 0 or startup_trials < 1 or trial_timeout <= 0:
        raise ValueError("Invalid budget, seed, startup_trials, or timeout")
    if time_budget is not None and time_budget <= 0:
        raise ValueError("time_budget must be positive")
    if warm_params and method != "tpe":
        raise ValueError("Warm start is supported only for TPE")
    effective_budget = min(budget, grid_size(space))
    if len(warm_params) > effective_budget or any(
        not validate_params(p, space) for p in warm_params
    ):
        raise ValueError("Warm-start queue must fit the budget and search space")
    measurement = measurement or Measurement()
    sampler = sampler_for(method, space, seed, startup_trials)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    start = time.perf_counter()
    metadata = {
        "workload": workload.signature(),
        "hardware": hardware.to_dict(),
        "search_space": space,
        "method": method,
        "seed": seed,
        "budget": effective_budget,
        "requested_budget": budget,
        "time_budget_s": time_budget,
        "trial_timeout_s": trial_timeout,
        "startup_trials": startup_trials,
        "heuristic_smem": heuristic_smem,
        "measurement": asdict(measurement),
        "transfer": transfer,
        "protocol": "cold_compile_per_candidate;cuda_graph;hot_reused_buffers",
        "code_fingerprint": code_fingerprint(),
        "environment": environment(),
    }
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="kernel",
        storage=f"sqlite:///{(output / 'study.sqlite3').resolve()}",
    )
    study.set_user_attr("metadata", metadata)
    for rank, params in enumerate(warm_params):
        study.enqueue_trial(params, user_attrs={"warm_start_rank": rank})
    seen = set()

    def objective(trial):
        trial_start = time.perf_counter()
        try:
            params = {
                name: trial.suggest_categorical(name, choices) for name, choices in space.items()
            }
            key = json.dumps(params, sort_keys=True)
            trial.set_user_attr("duplicate", key in seen)
            seen.add(key)
            trial.set_user_attr(
                "resource_estimate", resource_estimate(workload.kernel, params, workload.n)
            )
            reason = prune_reason(hardware, workload.kernel, params, workload.n, heuristic_smem)
            if reason:
                trial.set_user_attr("prune_reason", reason)
                raise optuna.TrialPruned(reason)
            remaining = trial_timeout
            if time_budget is not None:
                remaining = min(remaining, time_budget - (time.perf_counter() - start))
                if remaining <= 0:
                    raise EvaluationFailure("wall_time_budget_exhausted")
            result = (
                evaluator(params)
                if evaluator
                else isolated_evaluate(
                    workload,
                    hardware,
                    params,
                    measurement,
                    remaining,
                    output / "logs",
                    trial.number,
                )
            )
            value = float(result["latency_ms"])
            if not math.isfinite(value) or value <= 0:
                raise EvaluationFailure("invalid_latency")
            trial.set_user_attr("measurement", result)
            return value
        except EvaluationFailure as exc:
            trial.set_user_attr("failure", str(exc))
            raise
        finally:
            trial.set_user_attr("evaluation_wall_s", time.perf_counter() - trial_start)
            trial.set_user_attr("elapsed_s", time.perf_counter() - start)

    def checkpoint(study, trial):
        export(study, output, metadata, "running", time.perf_counter() - start)
        if time_budget is not None and time.perf_counter() - start >= time_budget:
            study.stop()

    status = "interrupted"
    try:
        # Grid exhaustion stops safely inside optimize; failed/pruned trials consume attempts.
        attempts = effective_budget
        remaining = (
            None if time_budget is None else max(0, time_budget - (time.perf_counter() - start))
        )
        study.optimize(
            objective,
            n_trials=attempts,
            timeout=remaining,
            n_jobs=1,
            catch=(EvaluationFailure,),
            callbacks=[checkpoint],
        )
        status = (
            "complete"
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            else "failed"
        )
    finally:
        data = export(study, output, metadata, status, time.perf_counter() - start)
    return data
