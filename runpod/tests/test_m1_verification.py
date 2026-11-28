"""Synthetic CPU archive tests; these fixtures are not CUDA performance evidence."""

import copy
import csv
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from bo_kernel_tuner.configs.search_spaces import Workload, grid_configs, search_space
from bo_kernel_tuner.tuner.worker import (
    BASELINES,
    MEASUREMENT_PROTOCOL,
    PRECISION_POLICY,
    Measurement,
    sample_statistics,
)
from runpod import paid_worker
from runpod.launch import GPU_TEST_MINIMUM, load_config, make_plan
from runpod.provider import SOURCE
from runpod.verify_m1 import CONFIRM_REPETITIONS
from runpod.verify_paid import ROOT, verify_archive


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(implementation, dtype, value, repeats=7):
    rtol, atol = (0.01, 0.02) if dtype == "fp16" else (0.02, 0.15)
    return {
        "status": "ok",
        "implementation": implementation,
        "protocol": MEASUREMENT_PROTOCOL,
        "precision_policy": PRECISION_POLICY,
        **sample_statistics([value] * repeats),
        "validation": [{"max_abs_error": 0.0, "rtol": rtol, "atol": atol}] * 2,
    }


class WorkerCommandTests(unittest.TestCase):
    def test_paid_worker_enables_exact_confirmation_plan(self):
        command = paid_worker.suite_command()
        self.assertEqual(command[2:4], ["runpod.launch", "smoke"])
        self.assertEqual(command[-2:], ["--confirm-repetitions", str(CONFIRM_REPETITIONS)])
        self.assertEqual(paid_worker.CONFIRM_REPETITIONS, 6)
        self.assertEqual(GPU_TEST_MINIMUM, 25)


@unittest.skipUnless(importlib.util.find_spec("optuna"), "M1 verification requires Optuna")
class ArchiveTests(unittest.TestCase):
    suite = "smoke"
    hardware_name = "SYNTHETIC-CPU-TEST-ONLY"

    def verify(self, archive, **kwargs):
        return verify_archive(archive, suite=self.suite, **kwargs)

    def setUp(self):
        from bo_kernel_tuner.tuner import confirmation as confirmation
        from bo_kernel_tuner.tuner.engine import code_fingerprint

        self.c = confirmation
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / ("faac-" + self.suite)
        self.output.mkdir()
        self.config = load_config(self.suite)
        self.plan = make_plan(self.config)
        hardware = {"uuid": "SYNTHETIC-CPU-TEST-ONLY", "device": 0, "name": self.hardware_name}
        versions = {"torch": "2.9.0+cu128", "triton": "3.5.0"}
        manifest = {
            "source_revision": SOURCE,
            "config": self.config,
            "sources": {},
            "code_fingerprint": code_fingerprint(),
            "runner_hash": sha(ROOT / "runpod/launch.py"),
            "hardware": hardware,
            "versions": versions,
            "confirmation_repetitions": 6,
        }
        write(self.output / "manifest.json", manifest)
        write(self.output / "plan.json", self.plan)
        write(
            self.output / "status.json",
            {
                "status": "complete",
                "gpu_validation_passed": True,
                "gpu_tests_passed": True,
                "completed_studies": len(self.plan),
                "completed_confirmations": len(self.plan),
            },
        )
        (self.output / "gpu-tests.xml").write_text(
            "<testsuite>" + '<testcase name="synthetic"/>' * 25 + "</testsuite>"
        )
        rows = []
        for item in self.plan:
            w = Workload(**item["case"]["workload"])
            space = search_space(w.kernel, item["case"]["profile"])
            measurement = Measurement(**self.config["measurement"])
            best = {
                "number": 0,
                "state": "COMPLETE",
                "value": 1.0,
                "params": next(grid_configs(space)),
                "attrs": {"measurement": timing("triton", w.dtype, 1.0, measurement.repeats)},
            }
            attempts = item["case"]["trials"] if self.suite == "pilot" else 1
            source = {
                "schema_version": 1,
                "status": "complete",
                "best": best,
                "trials": [{**copy.deepcopy(best), "number": i} for i in range(attempts)],
                "attempts": attempts,
                "successful_evaluations": attempts,
                "metadata": {
                    "workload": w.signature(),
                    "hardware": hardware,
                    "search_space": space,
                    "measurement": asdict(measurement),
                    "method": item["label"],
                    "seed": item["seed"],
                    "budget": item["case"]["trials"],
                    "code_fingerprint": code_fingerprint(),
                    "environment": {"versions": versions},
                    "precision_policy": PRECISION_POLICY,
                    "protocol": "cold_compile_per_candidate;" + MEASUREMENT_PROTOCOL,
                },
            }
            folder = self.output / item["path"]
            write(folder / "result.json", source)
            for name in ("study.sqlite3", "trials.csv"):
                (folder / name).write_text("synthetic CPU fixture, not a database/measurement")
            write(
                folder / "receipt.json",
                {
                    name: sha(folder / name)
                    for name in ("result.json", "study.sqlite3", "trials.csv")
                },
            )
            planned = self.c.make_plan(w, measurement, 6)
            pairs = [
                {
                    "status": "ok",
                    "order": p["order"],
                    "data_seed": p["data_seed"],
                    "hardware": hardware,
                    "candidate": timing("triton", w.dtype, 2.0, measurement.repeats),
                    "torch": timing(BASELINES[w.kernel], w.dtype, 4.0, measurement.repeats),
                }
                for p in planned
            ]
            summary = self.c.summarize(pairs)
            report = {
                "schema_version": 1,
                "status": "complete",
                "protocol": self.c.PROTOCOL,
                "source_sha256": sha(folder / "result.json"),
                "selected_params": best["params"],
                "search_best_ms": 1.0,
                "baseline": BASELINES[w.kernel],
                "plan": planned,
                "measurement": asdict(measurement),
                "code_fingerprint": code_fingerprint(),
                "precision_policy": PRECISION_POLICY,
                "hardware": hardware,
                "environment": {"versions": versions},
                "repetitions": 6,
                "pairs": pairs,
                "summary": summary,
            }
            destination = self.output / "confirmations" / item["path"]
            write(destination / "confirmation.json", report)
            (destination / "source-result.json").write_bytes((folder / "result.json").read_bytes())
            (destination / "pairs.csv").write_text(self.c.pairs_csv(pairs))
            self.receipt(destination)
            rows.append(
                {
                    "case": item["case"]["id"],
                    "method": item["label"],
                    "seed": item["seed"],
                    "baseline": report["baseline"],
                    "search_best_ms": 1.0,
                    **{k: v for k, v in summary.items() if k != "paired_speedups"},
                }
            )
        with (self.output / "baseline-summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for case in self.config["cases"]:
            folder = self.output / "figures" / case["id"]
            folder.mkdir(parents=True)
            for name in (
                "convergence.pdf",
                "tuning_time.pdf",
                "parameter_shift.pdf",
                "summary.csv",
                "paired_comparisons.csv",
            ):
                (folder / name).write_text("synthetic report fixture")

    def receipt(self, folder):
        write(folder / "receipt.json", {name: sha(folder / name) for name in self.c.FILES})

    def archive(self):
        # Recalculate outer checksums after tampering: semantic verification must
        # reject even an internally hash-consistent but invalid archive.
        write(
            self.output / "checksums.json",
            {
                str(p.relative_to(self.output)): sha(p)
                for p in self.output.rglob("*")
                if p.is_file() and p.name != "checksums.json"
            },
        )
        path = self.output.with_suffix(".tar.gz")
        with tarfile.open(path, "w:gz") as tar:
            tar.add(self.output, arcname=self.output.name)
        return path

    def test_full_m1_archive_is_recomputed_without_cuda(self):
        self.assertEqual(
            self.verify(self.archive()),
            {"confirmations": len(self.plan), "paired_workers": len(self.plan) * 6},
        )

    def test_incomplete_counts_are_rejected(self):
        for name, key, value in (
            ("status.json", "completed_confirmations", 17),
            ("manifest.json", "confirmation_repetitions", 0),
        ):
            with self.subTest(key=key):
                path = self.output / name
                original = path.read_bytes()
                data = json.loads(original)
                data[key] = value
                write(path, data)
                with self.assertRaisesRegex(RuntimeError, "confirmations"):
                    self.verify(self.archive())
                path.write_bytes(original)

    def test_missing_confirmation_is_rejected(self):
        path = self.output / "confirmations" / self.plan[0]["path"] / "confirmation.json"
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "confirmation set"):
            self.verify(self.archive())

    def test_source_cannot_be_substituted_by_another_valid_search(self):
        folder = self.output / "confirmations" / self.plan[0]["path"]
        other = self.output / "confirmations" / self.plan[1]["path"]
        for name in (*self.c.FILES, "receipt.json"):
            (folder / name).write_bytes((other / name).read_bytes())
        with self.assertRaisesRegex(RuntimeError, "bound to its accepted search"):
            self.verify(self.archive())

    def test_rehashed_invalid_pair_and_summary_rejected(self):
        folder = self.output / "confirmations" / self.plan[0]["path"]
        original = json.loads((folder / "confirmation.json").read_text())
        for change in ("seed", "summary", "partial"):
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                if change == "seed":
                    report["pairs"][0]["data_seed"] = 17
                elif change == "summary":
                    report["summary"]["median_paired_speedup"] = 999
                else:
                    report["pairs"].pop()
                write(folder / "confirmation.json", report)
                self.receipt(folder)
                with self.assertRaises(ValueError):
                    self.verify(self.archive())

    def test_valid_three_pair_confirmation_cannot_pass_six_pair_gate(self):
        folder = self.output / "confirmations" / self.plan[0]["path"]
        report = json.loads((folder / "confirmation.json").read_text())
        report.update(repetitions=3, plan=report["plan"][:3], pairs=report["pairs"][:3])
        report["summary"] = self.c.summarize(report["pairs"])
        write(folder / "confirmation.json", report)
        (folder / "pairs.csv").write_text(self.c.pairs_csv(report["pairs"]))
        self.receipt(folder)
        with self.assertRaisesRegex(RuntimeError, "repetition count"):
            self.verify(self.archive())

    def test_csv_cannot_publish_different_or_duplicate_results(self):
        path = self.output / "baseline-summary.csv"
        original = path.read_text()
        for text in (original.replace("2.0", "999.0"), original + original.splitlines()[1] + "\n"):
            with self.subTest(text=text[:30]):
                path.write_text(text)
                with self.assertRaisesRegex(RuntimeError, "Baseline summary"):
                    self.verify(self.archive())


if __name__ == "__main__":
    unittest.main()
