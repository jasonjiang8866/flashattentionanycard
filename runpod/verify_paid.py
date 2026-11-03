"""Independently validate downloaded CUDA evidence on the CPU Actions runner."""

import hashlib
import json
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from bo_kernel_tuner.configs.search_spaces import Workload, grid_size, search_space
from bo_kernel_tuner.tuner.worker import Measurement
from runpod.launch import load_config, make_plan, validate_junit, verify_study
from runpod.provider import SOURCE

ROOT = Path(__file__).resolve().parents[1]


def verify_archive(archive):
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        seen, size = set(), 0
        with tarfile.open(archive, "r:gz") as tar:
            for i, member in enumerate(tar):
                name = PurePosixPath(member.name)
                if (
                    i > 2000
                    or name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != "faac-smoke"
                    or member.name in seen
                    or not (member.isdir() or member.isfile())
                ):
                    raise RuntimeError("Unsafe result archive")
                seen.add(member.name)
                size += member.size
                if size > 256 * 1024 * 1024:
                    raise RuntimeError("Expanded archive exceeds limit")
                if member.isfile():
                    target = destination / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open("wb") as f:
                        while block := source.read(1024 * 1024):
                            f.write(block)
        output = destination / "faac-smoke"
        checksums = json.loads((output / "checksums.json").read_text())
        actual = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*")
            if p.is_file() and p.name != "checksums.json"
        }
        if not actual or actual != checksums:
            raise RuntimeError("Result member checksum mismatch")
        config = load_config("smoke")
        plan = make_plan(config)
        if json.loads((output / "plan.json").read_text()) != plan or len(plan) != 18:
            raise RuntimeError("Unexpected study plan")
        state = json.loads((output / "status.json").read_text())
        if (
            state.get("status") != "complete"
            or state.get("gpu_validation_passed") is not True
            or state.get("gpu_tests_passed") is not True
            or state.get("completed_studies") != 18
        ):
            raise RuntimeError("Incomplete GPU results")
        validate_junit(output / "gpu-tests.xml")
        manifest = json.loads((output / "manifest.json").read_text())
        fingerprint = hashlib.sha256()
        code_root = ROOT / "bo_kernel_tuner"
        for sub in ("kernels", "configs", "tuner"):
            for path in sorted((code_root / sub).glob("*.py")):
                fingerprint.update(str(path.relative_to(code_root)).encode())
                fingerprint.update(path.read_bytes())
        if (
            manifest.get("source_revision") != SOURCE
            or manifest.get("config") != config
            or manifest.get("sources") != {}
            or manifest.get("code_fingerprint") != fingerprint.hexdigest()
            or manifest.get("runner_hash")
            != hashlib.sha256((ROOT / "runpod/launch.py").read_bytes()).hexdigest()
        ):
            raise RuntimeError("Workload provenance mismatch")
        versions = manifest.get("versions", {})
        if versions.get("torch") != "2.9.0+cu128" or versions.get("triton") != "3.5.0":
            raise RuntimeError("Unexpected CUDA software versions")
        for item in plan:
            case = item["case"]
            w = Workload(**case["workload"])
            space = search_space(w.kernel, case["profile"])
            expected = {
                "workload": w.signature(),
                "hardware": manifest["hardware"],
                "search_space": space,
                "method": item["label"],
                "seed": item["seed"],
                "budget": min(case["trials"], grid_size(space)),
                "measurement": asdict(Measurement(**config["measurement"])),
                "code_fingerprint": fingerprint.hexdigest(),
            }
            verify_study(output / item["path"], expected)
        for case in config["cases"]:
            folder = output / "figures" / case["id"]
            for name in (
                "convergence.pdf",
                "tuning_time.pdf",
                "parameter_shift.pdf",
                "summary.csv",
                "paired_comparisons.csv",
            ):
                files = list(folder.rglob(name))
                if not files or any(not p.stat().st_size for p in files):
                    raise RuntimeError("Missing report")
