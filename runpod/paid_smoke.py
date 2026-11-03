"""One authorized $1 experiment; only this Actions controller has account credentials."""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from runpod.provider import (
    IMAGE,
    MIG,
    REPOSITORY,
    SOURCE,
    NoRedirect,
    Runpod,
    budget,
    finite,
    request,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runpod-results"
GPUS = (MIG, "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090")
OWNER = "FAAC_MANAGED_BY"
AUTHORIZATION = "faac-20261103-usd1"
# This authorization cannot be reused after artifact retention expires.
AUTHORIZATION_EXPIRES = 1794268800  # 2026-11-10 00:00 UTC
DISK_GB = 40
DISK_HOURLY = DISK_GB * 0.10 / (28 * 24)  # conservative 28-day billing month
CLEANUP_SECONDS = 360
MAX_WORK_SECONDS = 1200  # includes allocation, image pull, tests and result transfer
MAX_RATE = 1.50
MAX_ARCHIVE = 64 * 1024 * 1024
MARKER = "runpod-authorization-" + AUTHORIZATION


def save(name, value):
    OUT.mkdir(exist_ok=True)
    temporary = OUT / (name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUT / name)


def identity():
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Unexpected repository")
    run = os.environ.get("GITHUB_RUN_ID", "")
    if not run.isdigit():
        raise RuntimeError("Invalid Actions run ID")
    return run


def quote(rate):
    if not finite(rate, 0.001) or rate > MAX_RATE:
        raise ValueError("GPU rate missing or above $1.50/hour")
    hourly = rate + DISK_HOURLY
    work = min(MAX_WORK_SECONDS, math.floor(0.80 / hourly * 3600) - CLEANUP_SECONDS)
    if work < 300:
        raise ValueError("Insufficient useful runtime within budget")
    return {
        "gpu_hourly_usd": rate,
        "disk_hourly_usd": DISK_HOURLY,
        "work_seconds": work,
        "cleanup_seconds": CLEANUP_SECONDS,
        "planned_total_usd": hourly * (work + CLEANUP_SECONDS) / 3600,
        "approved_usd": 1,
        "unallocated_reserve_usd": 1 - hourly * (work + CLEANUP_SECONDS) / 3600,
        "transfer_usd": 0,
        "storage_gb": DISK_GB,
        "pricing_source": "https://docs.runpod.io/pods/pricing",
        "note": "Estimate, not a provider-enforced spending cap; deletion failure can overrun.",
    }


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def provenance():
    head = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[a-f0-9]{40}", head):
        raise RuntimeError("Invalid source revision")
    # The pinned image runs this exact workload; controller additions may differ.
    changed = git(
        "diff",
        "--name-only",
        SOURCE,
        head,
        "--",
        "bo_kernel_tuner",
        "runpod/launch.py",
        "runpod/configs",
        "runpod/requirements.txt",
        "runpod/install.sh",
        "pyproject.toml",
    )
    if changed:
        raise RuntimeError("Current workload differs from pinned image; review image/source pin")
    return {
        "controller_sha": head,
        "image_source_sha": SOURCE,
        "image": IMAGE,
        "suite": "smoke",
        "config_sha256": hashlib.sha256(
            (ROOT / "runpod/configs/smoke.json").read_bytes()
        ).hexdigest(),
        "worker_sha256": hashlib.sha256((ROOT / "runpod/paid_worker.py").read_bytes()).hexdigest(),
    }


def preflight(api):
    identity()
    evidence = provenance()
    pods = api.listing("/pods", "pods")
    if any(p.get("env", {}).get(OWNER) == REPOSITORY for p in pods):
        raise RuntimeError("An owned Pod already exists; cleanup and inspect before launch")
    choices, catalog = [], []
    for name in GPUS:
        try:
            data = api.call(
                "/catalog/gpus/"
                + urllib.parse.quote(name, safe="")
                + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8"
            )
            rate = data.get("price", {}).get("secure")
            cost = quote(rate)
            regions = sorted(
                d["id"]
                for d in data.get("dataCenters", [])
                if d.get("availability") in {"LOW", "MEDIUM", "HIGH"}
                and re.fullmatch(r"[A-Za-z0-9-]+", d.get("id", ""))
            )
            catalog.append({"gpu": name, "rate": rate, "regions": regions})
            if regions:
                choices.append((rate, name, regions, cost))
        except (RuntimeError, ValueError):
            catalog.append({"gpu": name, "status": "unavailable or price unverified"})
    if not choices:
        raise RuntimeError("No approved GPU with verified price/capacity")
    rate, name, regions, cost = min(choices)
    evidence.update(
        status="preflight_passed",
        read_only=True,
        catalog=catalog,
        gpu=name,
        regions=regions,
        cost=cost,
        budget=budget(api, reserve=max(1, rate + DISK_HOURLY)),
        checked_epoch=time.time(),
    )
    save("preflight.json", evidence)
    return evidence


def authorization_guard():
    run = identity()
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
        or os.environ.get("CONFIRM_PAID") != "true"
        or os.environ.get("EXPECTED_SHA") != git("rev-parse", "HEAD")
        or time.time() >= AUTHORIZATION_EXPIRES
    ):
        raise RuntimeError("Manual authorization/ref/SHA/attempt/expiry guard failed")
    # A persisted Actions artifact consumes this one-off authorization before POST /pods.
    # Re-dispatch and re-run cannot silently buy another experiment.
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise RuntimeError("Missing read-only Actions token")
    base = f"https://api.github.com/repos/{REPOSITORY}"
    remote = request(base + "/git/ref/heads/main", token=token)
    if remote.get("object", {}).get("sha") != os.environ["EXPECTED_SHA"]:
        raise RuntimeError("main moved; no paid launch")
    marker_seen = False
    for page in range(1, 101):
        data = request(
            base
            + "/actions/artifacts?"
            + urllib.parse.urlencode({"name": MARKER, "per_page": 100, "page": page}),
            token=token,
        )
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError("Cannot verify authorization ledger")
        for artifact in artifacts:
            if artifact.get("name") != MARKER:
                continue
            if str(artifact.get("workflow_run", {}).get("id")) != run:
                raise RuntimeError("This $1 authorization was already consumed; do not retry")
            marker_seen = True
        if len(artifacts) < 100:
            break
    else:
        raise RuntimeError("Incomplete authorization ledger")
    if not marker_seen:
        raise RuntimeError("Authorization marker must be uploaded before create")


def owned(pod, state):
    env = pod.get("env", {})
    return (
        pod.get("name") == state["name"]
        and env.get(OWNER) == REPOSITORY
        and env.get("FAAC_RUN_ID") == state["run_id"]
        and env.get("FAAC_AUTHORIZATION") == AUTHORIZATION
        and (not state.get("pod_id") or pod.get("id") == state["pod_id"])
    )


def cleanup(api):
    path = OUT / "provision.json"
    if not path.exists():
        # Fresh runner recovery also covers cancellation/loss of the original runner.
        run = identity()
        name = AUTHORIZATION + "-" + run
        matches = [p for p in api.listing("/pods", "pods") if p.get("name") == name]
        if not matches:
            save(
                "cleanup.json",
                {
                    "status": "verified_absent",
                    "name": name,
                    "checked_epoch": time.time(),
                    "cost": "unverified_no_local_intent",
                },
            )
            return
        if len(matches) != 1:
            raise RuntimeError("Ambiguous recovery inventory")
        pod = matches[0]
        state = {"name": name, "run_id": run, "pod_id": pod.get("id")}
        if not owned(pod, state):
            raise RuntimeError("Recovery ownership mismatch")
        state.update(
            started_epoch=float(pod["env"]["FAAC_STARTED"]),
            cost=quote(float(pod["env"]["FAAC_RATE"])),
        )
        save("provision.json", state)
    state = json.loads(path.read_text())
    # Reconcile by exact name even if POST returned an unknown outcome.
    for attempt in range(8):
        pods = api.listing("/pods", "pods")
        matches = [
            p
            for p in pods
            if p.get("name") == state["name"]
            or (state.get("pod_id") and p.get("id") == state["pod_id"])
        ]
        if not matches:
            # Require repeated independent inventories to cover delayed create visibility.
            if attempt >= 2 and state.get("pod_id"):
                elapsed = time.time() - state["started_epoch"]
                estimate = elapsed / 3600 * (state["cost"]["gpu_hourly_usd"] + DISK_HOURLY)
                save(
                    "cleanup.json",
                    {
                        "status": "verified_absent",
                        "name": state["name"],
                        "checked_epoch": time.time(),
                        "elapsed_billing_bound_s": elapsed,
                        "estimated_total_usd": estimate,
                        "within_budget": estimate <= 1,
                        "persistent_volumes_created": False,
                    },
                )
                if estimate > 1:
                    raise RuntimeError("Estimated spend exceeded authorization")
                return
        else:
            if len(matches) != 1 or not owned(matches[0], state):
                raise RuntimeError("Cleanup ownership mismatch; refusing unrelated deletion")
            pod_id = matches[0].get("id", "")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", pod_id):
                raise RuntimeError("Invalid Pod ID")
            if matches[0].get("mounts"):
                raise RuntimeError("Unexpected persistent mount; inspect before deletion")
            state["pod_id"] = pod_id
            save("provision.json", state)
            try:
                api.call("/pods/" + pod_id, "DELETE")
            except RuntimeError:
                pass  # DELETE outcome is reconciled by next inventory, never claimed successful.
        time.sleep(10)
    save("cleanup.json", {"status": "unknown", "name": state["name"]})
    raise RuntimeError("Owned Pod removal not verified; inspect immediately")


def download(url, token, expected):
    size, digest = expected.get("size"), expected.get("sha256")
    if (
        type(size) is not int
        or not 0 < size <= MAX_ARCHIVE
        or not re.fullmatch(r"[a-f0-9]{64}", digest or "")
    ):
        raise RuntimeError("Invalid result archive metadata")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    path = OUT / "smoke.tar.gz"
    hasher, count = hashlib.sha256(), 0
    with (
        urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response,
        path.open("wb") as f,
    ):
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > size:
                raise RuntimeError("Oversized result archive")
            hasher.update(block)
            f.write(block)
    if count != size or hasher.hexdigest() != digest:
        raise RuntimeError("Archive size/checksum mismatch")


class Deadline(BaseException):
    pass


def interrupt(signum, frame):
    raise Deadline("Work deadline or cancellation; begin cleanup")


def launch(api):
    authorization_guard()
    evidence = preflight(api)
    run = identity()
    token = secrets.token_urlsafe(32)
    print("::add-mask::" + token, flush=True)
    started = time.time()
    deadline = started + evidence["cost"]["work_seconds"]
    state = {
        "name": AUTHORIZATION + "-" + run,
        "run_id": run,
        "started_epoch": started,
        "deadline_epoch": deadline,
        "cost": evidence["cost"],
        "pod_id": None,
    }
    save("provision.json", state)  # durable intent BEFORE the only create request
    worker = base64.b64encode((ROOT / "runpod/paid_worker.py").read_bytes()).decode()
    code = (
        "import base64;exec(compile(base64.b64decode("
        + repr(worker)
        + "),'paid_worker.py','exec'))"
    )
    payload = {
        "name": state["name"],
        "image": IMAGE,
        "cloud": "SECURE",
        "disk": DISK_GB,
        "gpu": {
            "id": evidence["gpu"],
            "count": 1,
            "minRamPerGpu": 16,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": "12.8",
        },
        "dataCenterIds": evidence["regions"],
        "startSsh": False,
        "ports": ["8888/http"],
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": ["exec /opt/venv/bin/python -u -c " + shlex.quote(code)],
        "env": {
            OWNER: REPOSITORY,
            "FAAC_RUN_ID": run,
            "FAAC_AUTHORIZATION": AUTHORIZATION,
            "FAAC_RESULT_TOKEN": token,
            "FAAC_DEADLINE": str(deadline - 90),
            "FAAC_SOURCE_SHA": SOURCE,
            "FAAC_CONFIG_SHA": evidence["config_sha256"],
            "FAAC_CONTROLLER_SHA": evidence["controller_sha"],
            "FAAC_STARTED": str(started),
            "FAAC_RATE": str(evidence["cost"]["gpu_hourly_usd"]),
        },
    }
    handlers = {
        s: signal.signal(s, interrupt) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)
    }
    signal.alarm(evidence["cost"]["work_seconds"])
    try:
        pod = api.call("/pods", "POST", payload)  # never retried, including unknown outcome
        pod_id = pod.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pod_id):
            raise RuntimeError("Create returned no valid Pod ID; reconcile by name")
        state["pod_id"] = pod_id
        save("provision.json", state)
        detail = api.call("/pods/" + pod_id)
        if (
            not owned(detail, state)
            or detail.get("gpu", {}).get("id") != evidence["gpu"]
            or detail.get("dataCenterId") not in evidence["regions"]
            or not finite(detail.get("cost"), 0.001)
            or detail["cost"] > evidence["cost"]["gpu_hourly_usd"] + DISK_HOURLY
        ):
            raise RuntimeError("Allocated Pod identity, GPU, placement or rate mismatch")
        save(
            "allocation.json",
            {
                "id": pod_id,
                "gpu": evidence["gpu"],
                "data_center": detail["dataCenterId"],
                "observed_hourly_usd": detail["cost"],
            },
        )
        url = "https://" + pod_id + "-8888.proxy.runpod.net"
        while time.time() < deadline - 35:
            try:
                result = request(url + "/result", token=token)
            except RuntimeError:
                time.sleep(10)
                continue
            if result.get("status") in {"accepted", "rejected", "partial"}:
                # Worker returns only explicit evidence fields, never its environment.
                save("result.json", result)
                if result.get("archive"):
                    download(url + "/archive", token, result["archive"])
                if result.get("status") != "accepted":
                    raise RuntimeError("GPU experiment did not pass all acceptance gates")
                if (
                    result.get("source_sha") != SOURCE
                    or result.get("config_sha256") != evidence["config_sha256"]
                    or result.get("controller_sha") != evidence["controller_sha"]
                    or result.get("studies") != 18
                    or result.get("gpu_tests", 0) < 19
                    or not (OUT / "smoke.tar.gz").is_file()
                ):
                    raise RuntimeError("Result provenance/completeness mismatch")
                from runpod.verify_paid import verify_archive

                verify_archive(OUT / "smoke.tar.gz")
                save("acceptance.json", {"status": "accepted", "archive_verified": True})
                return
            time.sleep(10)
        raise RuntimeError("Deadline reached without complete artifacts")
    finally:
        signal.alarm(0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
        cleanup(api)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "cleanup"))
    args = parser.parse_args()
    identity()
    api = Runpod()
    if args.mode == "preflight":
        preflight(api)
    elif args.mode == "cleanup":
        cleanup(api)
    else:
        launch(api)


if __name__ == "__main__":
    main()
