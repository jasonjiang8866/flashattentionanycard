"""RunPod suite orchestration; scientific results remain owned by the core engine."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from bo_kernel_tuner.configs.search_spaces import Workload, grid_size, search_space
from bo_kernel_tuner.tuner.worker import Measurement

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(value):
    path = (
        ROOT / "runpod/configs" / f"{value}.json"
        if value in ("smoke", "pilot", "performance")
        else Path(value)
    )
    config = json.loads(path.read_text())
    if set(config) - {
        "schema_version",
        "seeds",
        "measurement",
        "trial_timeout",
        "time_budget",
        "cases",
    }:
        raise ValueError("Unknown suite configuration key")
    if config["schema_version"] != 1 or not config["cases"]:
        raise ValueError("Require schema_version=1 and at least one case")
    seeds = config["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    measurement = Measurement(**config["measurement"])
    if any(type(v) is not int for v in asdict(measurement).values()):
        raise ValueError("Measurement counts must be integers")
    for key in ("trial_timeout", "time_budget"):
        number = config.get(key)
        if key == "trial_timeout" and number is None:
            raise ValueError("trial_timeout is required")
        if number is not None and (
            type(number) not in (int, float) or not math.isfinite(number) or number <= 0
        ):
            raise ValueError(f"Invalid {key}")
    ids = set()
    for case in config["cases"]:
        if set(case) != {"id", "workload", "profile", "trials", "startup_trials"}:
            raise ValueError("Case requires id/workload/profile/trials/startup_trials")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case["id"]) or case["id"] in ids:
            raise ValueError("Case ids must be unique, safe directory names")
        ids.add(case["id"])
        w = Workload(**case["workload"])
        if any(type(v) is not int for v in (w.m, w.n, w.k, w.data_seed, *w.expert_tokens)):
            raise ValueError("Workload dimensions, data seed and token counts must be integers")
        space = search_space(w.kernel, case["profile"])
        if any(type(case[k]) is not int or case[k] < 1 for k in ("trials", "startup_trials")):
            raise ValueError("Trial budgets and startup counts must be positive integers")
        if case["startup_trials"] > min(case["trials"], grid_size(space)):
            raise ValueError("Startup trials exceed the effective candidate budget")
    return config


def make_plan(config, with_transfer=False):
    plan = []
    for case in config["cases"]:
        for seed in config["seeds"]:
            methods = ["grid", "random", "tpe"] + (["tpe_warm"] if with_transfer else [])
            random.Random(seed).shuffle(methods)
            for method in methods:
                plan.append(
                    {
                        "case": case,
                        "seed": seed,
                        "label": method,
                        "path": f"cases/{case['id']}/{method}-seed-{seed}",
                    }
                )
    return plan


def describe_plan(config, output, source_run=None, top_k=5):
    """CPU-only plan; candidate attempts include failures, pruning and duplicates."""
    plan = make_plan(config, source_run is not None)
    totals = {}
    for item in plan:
        case = item["case"]
        cardinality = grid_size(search_space(case["workload"]["kernel"], case["profile"]))
        budget = min(case["trials"], cardinality)
        if source_run is not None and top_k > budget:
            raise ValueError("--top-k exceeds a target case budget")
        item.update(
            max_attempts=budget,
            search_space_size=cardinality,
            output=str(output / item["path"]),
        )
        totals[item["label"]] = totals.get(item["label"], 0) + budget
    cap = sum(totals.values())
    return {
        "studies": len(plan),
        "max_attempts": cap,
        "max_attempts_by_method": totals,
        "output": str(output),
        "archive": str(output.parent / f"{output.name}.tar.gz"),
        "timing": str(output.parent / f"{output.name}.timing.json"),
        "candidate_worker_timeout_sum_s": cap * config["trial_timeout"],
        "time_limit_note": "Worker timeout sum only; excludes preflight, GPU tests, search/IO, "
        "analysis and archiving. Not a suite wall-time or billing cap.",
        "source_validation": "deferred until execution" if source_run is not None else None,
        "plan": plan,
    }


def validate_output(output, source_run, resume):
    """Check paths without creating directories or probing CUDA."""
    if source_run is not None:
        source = source_run.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output and source-run directories must not overlap")
    if output.exists():
        if not output.is_dir():
            raise ValueError("Output must be a directory")
        if any(output.iterdir()) and not resume:
            raise ValueError("Output exists; use a new --output or explicitly --resume")
    elif resume:
        raise ValueError("Cannot resume a missing output directory")
    for ancestor in output.parents:
        if ancestor.exists():
            if not ancestor.is_dir():
                raise ValueError("Output parent must be a directory")
            break
    for suffix in (".tar.gz", ".tar.tmp", ".timing.json", ".timing.tmp"):
        sidecar = output.parent / f"{output.name}{suffix}"
        if sidecar.exists() and (sidecar.is_dir() or not resume):
            raise ValueError(f"Output sidecar exists: {sidecar}")


def gpu_preflight(device):
    import torch
    import triton

    from bo_kernel_tuner.tuner.hardware import Hardware

    if (
        torch.__version__ != "2.9.0+cu128"
        or torch.version.cuda != "12.8"
        or triton.__version__ != "3.5.0"
    ):
        raise RuntimeError(
            "RunPod environment requires torch 2.9.0+cu128 and Triton 3.5.0; run bootstrap.sh"
        )
    if os.environ.get("TRITON_INTERPRET", "0") != "0":
        raise RuntimeError("Disable TRITON_INTERPRET for real GPU validation")
    hardware = Hardware.detect(device)
    torch.cuda.set_device(device)
    probe = torch.ones(8, device=f"cuda:{device}") * 2
    torch.cuda.synchronize(device)
    if probe.sum().item() != 16:
        raise RuntimeError("CUDA preflight failed")
    return hardware


def run_logged(command, log_path, env=None, timeout=1200):
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        expired = threading.Event()

        def terminate_timeout():
            expired.set()
            child.kill()

        timer = threading.Timer(timeout, terminate_timeout)
        timer.start()
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
            if expired.is_set():
                raise RuntimeError(f"Command exceeded {timeout}s; see {log_path.name}")
            if code:
                raise RuntimeError(f"Command exited {code}; see {log_path.name}")
        finally:
            timer.cancel()
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def validate_junit(path):
    root = ET.parse(path).getroot()
    tests = list(root.iter("testcase"))
    if len(tests) < 19 or any(
        t.find(tag) is not None for t in tests for tag in ("failure", "error", "skipped")
    ):
        raise RuntimeError(
            "GPU tests did not fully pass (need >=19 executed tests, zero skips/errors)"
        )
    return len(tests)


def gpu_tests(output, device):
    # Tests use cuda:0 internally; remap the requested logical device to physical UUID.
    from bo_kernel_tuner.tuner.hardware import Hardware

    hardware = Hardware.detect(device)
    if hardware.uuid == "unknown":
        raise RuntimeError("GPU UUID unavailable; cannot reliably select the device for tests")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=hardware.uuid)
    run_logged(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "gpu",
            "bo_kernel_tuner/tests/test_correctness.py",
            f"--junitxml={output / 'gpu-tests.xml'}",
        ],
        output / "gpu-tests.log",
        env,
    )
    return validate_junit(output / "gpu-tests.xml")


def expected_metadata(item, config, hardware):
    from bo_kernel_tuner.tuner.engine import code_fingerprint

    case = item["case"]
    w = Workload(**case["workload"])
    space = search_space(w.kernel, case["profile"])
    return {
        "workload": w.signature(),
        "hardware": hardware.to_dict(),
        "search_space": space,
        "method": "tpe" if item["label"] == "tpe_warm" else item["label"],
        "seed": item["seed"],
        "budget": min(case["trials"], grid_size(space)),
        "measurement": asdict(Measurement(**config["measurement"])),
        "code_fingerprint": code_fingerprint(),
    }


def verify_study(directory, expected, require_receipt=True):
    required = [directory / name for name in ("result.json", "trials.csv", "study.sqlite3")]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise RuntimeError(f"Incomplete study artifacts: {directory}")
    result = json.loads(required[0].read_text())
    if result.get("schema_version") != 1 or result.get("status") != "complete":
        raise RuntimeError(f"Study not complete: {directory}")
    if any(result["metadata"].get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Study protocol mismatch: {directory}")
    trials = result["trials"]
    successful = [t for t in trials if t["state"] == "COMPLETE"]
    if (
        not successful
        or not 0 < len(trials) <= expected["budget"]
        or result["attempts"] != len(trials)
    ):
        raise RuntimeError("Invalid study attempt accounting")
    if result["successful_evaluations"] != len(successful):
        raise RuntimeError("Invalid successful trial count")
    for t in successful:
        measured = t["attrs"].get("measurement", {})
        samples = measured.get("samples_ms", [])
        validation = measured.get("validation", [])
        if (
            measured.get("status") != "ok"
            or len(samples) != expected["measurement"]["repeats"]
            or any(not math.isfinite(v) or v <= 0 for v in samples)
            or not math.isfinite(t["value"])
            or t["value"] <= 0
            or not math.isclose(t["value"], statistics.median(samples), rel_tol=1e-8)
            or len(validation) != expected["measurement"]["validation_seeds"]
            or any(not math.isfinite(v.get("max_abs_error", float("inf"))) for v in validation)
        ):
            raise RuntimeError("Incomplete/invalid GPU measurement evidence")
    if result["best"] != min(successful, key=lambda t: t["value"]):
        raise RuntimeError("Best trial does not match successful measurements")
    hashes = {p.name: sha256(p) for p in required}
    receipt = directory / "receipt.json"
    if require_receipt and (not receipt.is_file() or json.loads(receipt.read_text()) != hashes):
        raise RuntimeError(f"Study receipt missing or changed: {directory}")
    return hashes


def source_candidates(args, config, hardware):
    from bo_kernel_tuner.tuner.hardware import prune_reason
    from bo_kernel_tuner.tuner.transfer import load_warm_start

    candidates, sources = {}, {}
    if not args.source_run:
        return candidates, sources
    for case in config["cases"]:
        w = Workload(**case["workload"])
        space = search_space(w.kernel, case["profile"])
        if args.top_k > min(case["trials"], grid_size(space)):
            raise ValueError("--top-k exceeds a target case budget")
        path = (
            args.source_run / "cases" / case["id"] / f"tpe-seed-{args.source_seed}" / "result.json"
        )
        raw = json.loads(path.read_text())
        verify_study(
            path.parent,
            {
                k: raw["metadata"][k]
                for k in (
                    "workload",
                    "hardware",
                    "search_space",
                    "method",
                    "seed",
                    "budget",
                    "measurement",
                    "code_fingerprint",
                )
            },
        )
        if raw["metadata"]["measurement"] != asdict(Measurement(**config["measurement"])):
            raise ValueError("Source/target measurement settings differ")
        candidates[case["id"]] = load_warm_start(
            path,
            w,
            space,
            args.top_k,
            hardware,
            lambda p, w=w: prune_reason(hardware, w.kernel, p, w.n),
        )
        sources[case["id"]] = {
            "sha256": sha256(path),
            "top_k": args.top_k,
            "source_seed": args.source_seed,
        }
    return candidates, sources


def run_context(args, config):
    from bo_kernel_tuner.tuner.engine import code_fingerprint, environment

    hardware = gpu_preflight(args.device)
    candidates, sources = source_candidates(args, config, hardware)
    env = environment()
    revision_file = ROOT / "SOURCE_REVISION"
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    else:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "config": config,
        "hardware": hardware.to_dict(),
        "code_fingerprint": code_fingerprint(),
        "runner_hash": sha256(ROOT / "runpod/launch.py"),
        "versions": env["versions"],
        "python": sys.version,
        "source_revision": revision,
        "sources": sources,
    }
    return hardware, candidates, env, manifest


def bundle(output):
    paths = sorted(
        p
        for p in output.rglob("*")
        if p.is_file() and p.name not in (".run.lock", "checksums.json")
    )
    write_json(output / "checksums.json", {str(p.relative_to(output)): sha256(p) for p in paths})
    archive = output.parent / f"{output.name}.tar.gz"
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        tar.add(
            output,
            arcname=output.name,
            filter=lambda info: None if info.name.endswith("/.run.lock") else info,
        )
    temporary.replace(archive)
    return archive


def execute(args, config, output):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    from bo_kernel_tuner.analysis.plot_metrics import generate
    from bo_kernel_tuner.tuner.engine import run_search

    output.mkdir(parents=True, exist_ok=True)
    # flock prevents two suite processes from writing the same experiment.
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = [p for p in output.iterdir() if p.name != ".run.lock"]
        if existing and not args.resume:
            raise ValueError("Output exists; use a new --output or explicitly --resume")
        context = None
        manifest_path = output / "manifest.json"
        # Reject mismatched resumes without changing the existing experiment's status or files.
        if existing:
            if not manifest_path.exists():
                raise ValueError("Existing output lacks a manifest; use a new output")
            context = run_context(args, config)
            if json.loads(manifest_path.read_text()) != context[3]:
                raise ValueError(
                    "Resume configuration/code/environment/hardware mismatch; use a new output"
                )
        timing_path = output.parent / f"{output.name}.timing.json"
        sessions = json.loads(timing_path.read_text())["sessions"] if timing_path.exists() else []
        state = {
            "status": "running",
            "phase": "preflight",
            "gpu_validation_passed": False,
            "gpu_tests_passed": False,
            "completed_studies": 0,
        }
        write_json(output / "status.json", state)
        try:
            hardware, candidates, env, manifest = context or run_context(args, config)
            write_json(manifest_path, manifest)
            write_json(output / "environment.json", env)
            run_logged([sys.executable, "-m", "pip", "freeze"], output / "pip-freeze.txt")
            state["phase"] = "gpu_correctness"
            write_json(output / "status.json", state)
            state["gpu_test_count"] = gpu_tests(output, args.device)
            state["gpu_tests_passed"] = True
            plan = make_plan(config, bool(candidates))
            write_json(output / "plan.json", plan)
            for item in plan:
                destination = output / item["path"]
                expected = expected_metadata(item, config, hardware)
                state["phase"] = item["path"]
                write_json(output / "status.json", state)
                if destination.exists():
                    old = destination / "result.json"
                    if (
                        old.exists()
                        and json.loads(old.read_text()).get("status") == "complete"
                        and (destination / "receipt.json").is_file()
                    ):
                        verify_study(destination, expected)
                        state["completed_studies"] += 1
                        print(f"Verified; skipping {item['path']}", flush=True)
                        continue
                    # Never append to a partially observed study with reset sampler state.
                    archived = output / "incomplete" / uuid.uuid4().hex / item["path"]
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(archived))
                case = item["case"]
                warm, provenance = (
                    candidates[case["id"]] if item["label"] == "tpe_warm" else ((), None)
                )
                print(f"Running {item['path']}", flush=True)
                run_search(
                    Workload(**case["workload"]),
                    hardware,
                    expected["search_space"],
                    expected["method"],
                    item["seed"],
                    case["trials"],
                    destination,
                    Measurement(**config["measurement"]),
                    config.get("time_budget"),
                    config["trial_timeout"],
                    case["startup_trials"],
                    warm,
                    provenance,
                )
                hashes = verify_study(destination, expected, require_receipt=False)
                write_json(destination / "receipt.json", hashes)
                state["completed_studies"] += 1
            state["phase"] = "analysis"
            write_json(output / "status.json", state)
            for case in config["cases"]:
                figure_dir = output / "figures" / case["id"]
                generate([output / "cases" / case["id"]], figure_dir)
                for name in (
                    "convergence.pdf",
                    "tuning_time.pdf",
                    "parameter_shift.pdf",
                    "summary.csv",
                    "paired_comparisons.csv",
                ):
                    files = list(figure_dir.rglob(name))
                    if not files or any(p.stat().st_size == 0 for p in files):
                        raise RuntimeError(f"Missing figure/report: {case['id']}/{name}")
            # Recheck all receipts at completion; status is never inferred just from exit codes.
            for item in plan:
                verify_study(output / item["path"], expected_metadata(item, config, hardware))
            state.update(status="complete", phase="complete", gpu_validation_passed=True)
        except BaseException as exc:
            state.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                gpu_validation_passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            write_json(output / "status.json", state)
            try:
                archive = bundle(output)
                print(f"Results: {output}\nArchive: {archive}", flush=True)
            except Exception as exc:
                state.update(status="failed", gpu_validation_passed=False, archive_error=str(exc))
                write_json(output / "status.json", state)
                raise
            finally:
                # A sidecar avoids changing archive members after their checksums were sealed.
                # Include preflight, validation, search, reports, checksums and bundle creation.
                sessions.append(
                    {
                        "started_at": started_at,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_s": time.perf_counter() - started,
                        "status": state["status"],
                        "resume": args.resume,
                    }
                )
                write_json(
                    timing_path,
                    {
                        "schema_version": 1,
                        "sessions": sessions,
                        "total_active_wall_s": sum(s["elapsed_s"] for s in sessions),
                        "scope": "Includes bundle creation; excludes idle time between resumes "
                        "and this final timing-file write. SIGKILL/power loss may omit a session.",
                    },
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        nargs="?",
        default=os.environ.get("BO_SUITE", "smoke"),
        help="smoke, pilot, performance, or custom suite JSON path",
    )
    parser.add_argument("--device", type=int, default=int(os.environ.get("BO_DEVICE", "0")))
    parser.add_argument("--output", type=Path, default=os.environ.get("BO_OUTPUT_DIR"))
    parser.add_argument("--source-run", type=Path, default=os.environ.get("BO_SOURCE_DIR"))
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("BO_TOP_K", "5")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.device < 0 or args.top_k < 1 or args.source_seed < 0:
            raise ValueError("Invalid device, top-k, or source seed")
        config = load_config(args.suite)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = (
            args.output
            or Path(os.environ.get("BO_WORKSPACE", "/workspace/bo-kernel-tuner")) / "results" / name
        ).resolve()
        validate_output(output, args.source_run, args.resume)
        plan = describe_plan(config, output, args.source_run, args.top_k)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        execute(args, config, output)
        return 0
    except KeyboardInterrupt:
        print(
            "Interrupted; partial results retained. Use --resume with the same output.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"RunPod suite failed: {exc}", file=sys.stderr)
        return 1


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
