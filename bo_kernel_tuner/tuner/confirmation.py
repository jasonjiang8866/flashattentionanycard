"""Freeze a search winner, then confirm it against PyTorch in fresh paired workers.

CUDA samples within a worker are not independent repetitions. Only complete pairs
enter the descriptive bootstrap; failed/partial runs never publish a speedup.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from bo_kernel_tuner.configs.search_spaces import Workload, validate_params
from bo_kernel_tuner.tuner.engine import (
    _atomic_json,
    code_fingerprint,
    environment,
    isolated_evaluate,
)
from bo_kernel_tuner.tuner.hardware import Hardware
from bo_kernel_tuner.tuner.worker import (
    BASELINES,
    MEASUREMENT_PROTOCOL,
    PRECISION_POLICY,
    Measurement,
    sample_statistics,
)

PROTOCOL = "paired_confirmation_v1;fresh_process_per_pair;held_out_inputs;balanced_order"
FILES = ("source-result.json", "confirmation.json", "pairs.csv")


def validate_timing(result, measurement, dtype, implementation):
    samples = result.get("samples_ms", [])
    summary = sample_statistics(samples)
    tolerance = (0.01, 0.02) if dtype == "fp16" else (0.02, 0.15)
    validation = result.get("validation", [])
    if (
        result.get("status") != "ok"
        or result.get("implementation") != implementation
        or result.get("protocol") != MEASUREMENT_PROTOCOL
        or result.get("precision_policy") != PRECISION_POLICY
        or len(samples) != measurement.repeats
        or any(result.get(k) != v for k, v in summary.items())
        or len(validation) != measurement.validation_seeds
        or any(
            type(v.get("max_abs_error")) not in (int, float)
            or not math.isfinite(v["max_abs_error"])
            or v["max_abs_error"] < 0
            or (v.get("rtol"), v.get("atol")) != tolerance
            for v in validation
        )
    ):
        raise ValueError("Incomplete or inconsistent confirmation measurement")


def freeze_source(raw, require_current=True):
    source = json.loads(raw)
    if source.get("schema_version") != 1 or source.get("status") != "complete":
        raise ValueError("Confirmation requires a completed schema-v1 search")
    meta = source["metadata"]
    workload = Workload(**meta["workload"])
    measurement = Measurement(**meta["measurement"])
    if (
        meta.get("protocol") != "cold_compile_per_candidate;" + MEASUREMENT_PROTOCOL
        or meta.get("precision_policy") != PRECISION_POLICY
        or (require_current and meta["code_fingerprint"] != code_fingerprint())
    ):
        raise ValueError("Source code/measurement protocol differs; run a new search")
    trials = source["trials"]
    completed = [t for t in trials if t["state"] == "COMPLETE"]
    if (
        not completed
        or source["attempts"] != len(trials)
        or source["successful_evaluations"] != len(completed)
        or any(
            type(t["value"]) not in (int, float) or not math.isfinite(t["value"]) or t["value"] <= 0
            for t in completed
        )
    ):
        raise ValueError("Invalid source trial accounting or latency")
    best = min(completed, key=lambda t: t["value"])
    if source["best"] != best or not validate_params(best["params"], meta["search_space"]):
        raise ValueError("Source winner does not match the successful trials")
    validate_timing(best["attrs"]["measurement"], measurement, workload.dtype, "triton")
    if best["value"] != best["attrs"]["measurement"]["latency_ms"]:
        raise ValueError("Source winner latency differs from its samples")
    return source, workload, measurement


def make_plan(workload, measurement, repetitions):
    if type(repetitions) is not int or repetitions < 3:
        raise ValueError("Require at least three independent paired repetitions")
    # Never reuse any search validation seed. Pairs also have disjoint seed ranges.
    return [
        {
            "repetition": i,
            "data_seed": workload.data_seed + (i + 1) * measurement.validation_seeds,
            "order": ["candidate", "torch"] if i % 2 == 0 else ["torch", "candidate"],
        }
        for i in range(repetitions)
    ]


def comparable_hardware(hardware):
    return {k: v for k, v in hardware.items() if k != "device"}


def validate_pair(pair, planned, hardware, workload, measurement):
    if (
        pair.get("status") != "ok"
        or any(pair.get(k) != planned[k] for k in ("order", "data_seed"))
        or comparable_hardware(pair.get("hardware", {})) != comparable_hardware(hardware)
    ):
        raise ValueError("Paired worker identity/order/seed mismatch")
    for label, implementation in (("candidate", "triton"), ("torch", BASELINES[workload.kernel])):
        validate_timing(pair[label], measurement, workload.dtype, implementation)


def summarize(pairs):
    ratios = [p["torch"]["latency_ms"] / p["candidate"]["latency_ms"] for p in pairs]
    rng = random.Random(0)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))

    def percentile(q):
        index = (len(bootstrap) - 1) * q
        low = math.floor(index)
        return bootstrap[low] + (bootstrap[math.ceil(index)] - bootstrap[low]) * (index - low)

    return {
        "paired_repetitions": len(pairs),
        "candidate_median_ms": statistics.median(p["candidate"]["latency_ms"] for p in pairs),
        "baseline_median_ms": statistics.median(p["torch"]["latency_ms"] for p in pairs),
        "median_paired_speedup": statistics.median(ratios),
        "paired_speedups": ratios,
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "interval_method": "descriptive_percentile_bootstrap_of_pairs;5000_draws;seed=0",
        "scope": "One frozen configuration on one device/session, not search-seed or fleet variance",
    }


def pairs_csv(pairs):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["repetition", "data_seed", "first", "candidate_ms", "baseline_ms", "speedup"])
    for i, pair in enumerate(pairs):
        candidate, baseline = pair["candidate"]["latency_ms"], pair["torch"]["latency_ms"]
        writer.writerow(
            [i, pair["data_seed"], pair["order"][0], candidate, baseline, baseline / candidate]
        )
    return stream.getvalue()


def verify_confirmation(output):
    """CPU-only recomputation of frozen selection, pair evidence, summary and receipt."""
    output = Path(output)
    raw = (output / "source-result.json").read_bytes()
    source, workload, measurement = freeze_source(raw, require_current=False)
    report = json.loads((output / "confirmation.json").read_text())
    expected_plan = make_plan(workload, measurement, report["repetitions"])
    if (
        report.get("schema_version") != 1
        or report.get("status") != "complete"
        or report.get("protocol") != PROTOCOL
        or report.get("source_sha256") != hashlib.sha256(raw).hexdigest()
        or report.get("selected_params") != source["best"]["params"]
        or report.get("search_best_ms") != source["best"]["value"]
        or report.get("baseline") != BASELINES[workload.kernel]
        or report.get("plan") != expected_plan
        or report.get("measurement") != asdict(measurement)
        or report.get("code_fingerprint") != source["metadata"]["code_fingerprint"]
        or report.get("precision_policy") != PRECISION_POLICY
        or comparable_hardware(report["hardware"])
        != comparable_hardware(source["metadata"]["hardware"])
        or report["hardware"].get("uuid", "unknown") == "unknown"
        or report["environment"]["versions"] != source["metadata"]["environment"]["versions"]
        or len(report["pairs"]) != len(expected_plan)
    ):
        raise ValueError("Confirmation provenance/completeness mismatch")
    for pair, planned in zip(report["pairs"], expected_plan):
        validate_pair(pair, planned, report["hardware"], workload, measurement)
    if report.get("summary") != summarize(report["pairs"]):
        raise ValueError("Confirmation summary differs from paired evidence")
    if (output / "pairs.csv").read_text() != pairs_csv(report["pairs"]):
        raise ValueError("Confirmation CSV differs from paired evidence")
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in FILES}
    if json.loads((output / "receipt.json").read_text()) != hashes:
        raise ValueError("Confirmation receipt mismatch")
    return report["summary"]


def confirm(source_path, output, device=0, repetitions=6, timeout=180, dry_run=False):
    raw = Path(source_path).read_bytes()
    source, workload, measurement = freeze_source(raw)
    plan = make_plan(workload, measurement, repetitions)
    if type(device) is not int or device < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid device or worker timeout")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new confirmation output directory")
    if dry_run:
        return {
            "baseline": BASELINES[workload.kernel],
            "plan": plan,
            "worker_timeout_sum_s": repetitions * timeout,
        }
    hardware = Hardware.detect(device)
    env = environment()
    if (
        hardware.uuid == "unknown"
        or comparable_hardware(hardware.to_dict())
        != comparable_hardware(source["metadata"]["hardware"])
        or env["versions"] != source["metadata"]["environment"]["versions"]
    ):
        raise ValueError("Confirmation requires the source GPU identity and software versions")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-result.json").write_bytes(raw)
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": PROTOCOL,
        "code_fingerprint": code_fingerprint(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_params": source["best"]["params"],
        "search_best_ms": source["best"]["value"],
        "baseline": BASELINES[workload.kernel],
        "hardware": hardware.to_dict(),
        "environment": env,
        "precision_policy": PRECISION_POLICY,
        "measurement": asdict(measurement),
        "repetitions": repetitions,
        "worker_timeout_s": timeout,
        "plan": plan,
        "pairs": [],
        "summary": None,
    }
    start = time.perf_counter()
    try:
        _atomic_json(output / "confirmation.json", report)
        for planned in plan:
            pair = isolated_evaluate(
                replace(workload, data_seed=planned["data_seed"]),
                hardware,
                report["selected_params"],
                measurement,
                timeout,
                output / "logs",
                planned["repetition"],
                order=planned["order"],
            )
            validate_pair(pair, planned, hardware.to_dict(), workload, measurement)
            report["pairs"].append(pair)
            _atomic_json(output / "confirmation.json", report)
        report.update(status="complete", summary=summarize(report["pairs"]))
        (output / "pairs.csv").write_text(pairs_csv(report["pairs"]))
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            summary=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - start
        _atomic_json(output / "confirmation.json", report)
    _atomic_json(
        output / "receipt.json",
        {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in FILES},
    )
    verify_confirmation(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="Completed search result.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path, help="Verify a confirmation directory without CUDA")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            if args.source or args.output or args.dry_run:
                raise ValueError("--verify cannot be combined with execution inputs")
            result = verify_confirmation(args.verify)
        else:
            if args.source is None or args.output is None:
                raise ValueError("source and --output are required")
            result = confirm(
                args.source, args.output, args.device, args.repetitions, args.timeout, args.dry_run
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"Confirmation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
