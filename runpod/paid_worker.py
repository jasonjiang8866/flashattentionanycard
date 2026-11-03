"""Injected into the pinned image; no RunPod or GitHub account credentials."""

import base64
import hashlib
import hmac
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/opt/bo-kernel-tuner")
OUTPUT = Path("/tmp/faac-smoke")
ARCHIVE = OUTPUT.with_suffix(".tar.gz")
RESULT = {"status": "running"}
TOKEN = ""


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN):
            self.send_error(403)
            return
        if self.path == "/result":
            data = json.dumps(RESULT, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/archive" and RESULT.get("archive"):
            self.send_response(200)
            self.send_header("Content-Length", str(ARCHIVE.stat().st_size))
            self.end_headers()
            with ARCHIVE.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    self.wfile.write(block)
        else:
            self.send_error(404)


def work():
    global RESULT
    source = os.environ["FAAC_SOURCE_SHA"]
    config_hash = os.environ["FAAC_CONFIG_SHA"]
    result = {
        "status": "rejected",
        "source_sha": source,
        "config_sha256": config_hash,
        "controller_sha": os.environ["FAAC_CONTROLLER_SHA"],
    }
    child = None
    try:
        if (ROOT / "SOURCE_REVISION").read_text().strip() != source:
            raise RuntimeError("image_revision_mismatch")
        if digest(ROOT / "runpod/configs/smoke.json") != config_hash:
            raise RuntimeError("configuration_mismatch")
        remaining = float(os.environ["FAAC_DEADLINE"]) - time.time()
        if remaining < 60:
            raise RuntimeError("startup_consumed_budget")
        # Exclusive marker prevents container restarts from rerunning the experiment.
        with Path("/tmp/faac-attempt-started").open("x") as f:
            f.write(source)
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in {"FAAC_RESULT_TOKEN", "RUNPOD_API_KEY", "GH_TOKEN", "GITHUB_TOKEN"}
        }
        with Path("/tmp/faac-suite.log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, "-m", "runpod.launch", "smoke", "--output", str(OUTPUT)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                result["status"] = "partial"
                raise RuntimeError("wall_time_limit") from None
        if code:
            raise RuntimeError("suite_failed")
        sys.path.insert(0, str(ROOT))
        from bo_kernel_tuner.tuner.hardware import Hardware
        from runpod.launch import (
            expected_metadata,
            load_config,
            make_plan,
            validate_junit,
            verify_study,
        )

        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        config = load_config("smoke")
        if manifest["source_revision"] != source or manifest["config"] != config:
            raise RuntimeError("manifest_mismatch")
        hardware = Hardware(**manifest["hardware"])
        plan = make_plan(config)
        for item in plan:
            verify_study(OUTPUT / item["path"], expected_metadata(item, config, hardware))
        tests = validate_junit(OUTPUT / "gpu-tests.xml")
        state = json.loads((OUTPUT / "status.json").read_text())
        if (
            state["status"] != "complete"
            or not state["gpu_validation_passed"]
            or state["completed_studies"] != len(plan)
        ):
            raise RuntimeError("incomplete_studies")
        result.update(
            status="accepted",
            studies=len(plan),
            gpu_tests=tests,
            hardware=manifest["hardware"],
            versions=manifest["versions"],
        )
    except Exception as exc:
        # Fixed exception type only; paths, tokens and raw environment never returned.
        result["error_type"] = type(exc).__name__
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        if ARCHIVE.is_file() and 0 < ARCHIVE.stat().st_size <= 64 * 1024 * 1024:
            result["archive"] = {"size": ARCHIVE.stat().st_size, "sha256": digest(ARCHIVE)}
            if ARCHIVE.stat().st_size <= 128 * 1024:
                result["archive_inline_base64"] = base64.b64encode(ARCHIVE.read_bytes()).decode()
        # Keep failure diagnostics even if the separate binary download endpoint fails.
        status_file = OUTPUT / "status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text())
            result["suite_status"] = {
                key: status[key]
                for key in ("status", "phase", "error", "gpu_test_count", "completed_studies")
                if key in status
            }
        timing = OUTPUT.with_suffix(".timing.json")
        if timing.is_file():
            result["timing"] = json.loads(timing.read_text())
        RESULT = result


def main():
    global TOKEN
    TOKEN = os.environ.pop("FAAC_RESULT_TOKEN")
    server = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    threading.Thread(target=work, daemon=True).start()
    # Controller deletes the Pod. Exiting alone does not prove billing stopped.
    server.serve_forever()


if __name__ == "__main__":
    main()
