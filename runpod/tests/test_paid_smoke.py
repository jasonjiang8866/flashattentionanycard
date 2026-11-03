import base64
import hashlib
import io
import json
import os
import tarfile
import tempfile
import time
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import Mock, patch

from runpod import paid_smoke as c
from runpod.provider import REPOSITORY, SOURCE, Runpod
from runpod.verify_paid import verify_archive


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.patches = [
            patch.object(c, "OUT", self.out),
            patch.dict(
                os.environ,
                {
                    "GITHUB_REPOSITORY": REPOSITORY,
                    "GITHUB_RUN_ID": "1234",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_EVENT_NAME": "workflow_dispatch",
                    "GITHUB_REF": "refs/heads/main",
                    "EXPECTED_SHA": "a" * 40,
                    "CONFIRM_PAID": "true",
                    "GH_TOKEN": "test-only",
                },
            ),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def state(self):
        return {
            "name": c.AUTHORIZATION + "-1234",
            "run_id": "1234",
            "pod_id": "pod1",
            "started_epoch": time.time(),
            "cost": c.quote(0.5),
        }

    def pod(self):
        return {
            "name": c.AUTHORIZATION + "-1234",
            "id": "pod1",
            "env": {
                c.OWNER: REPOSITORY,
                "FAAC_RUN_ID": "1234",
                "FAAC_AUTHORIZATION": c.AUTHORIZATION,
                "FAAC_STARTED": str(time.time()),
                "FAAC_RATE": ".5",
            },
        }

    def test_failed_preflight_preserves_capacity_evidence(self):
        api = Mock()
        api.listing.return_value = []
        api.call.return_value = {"price": {"secure": 0.69}, "dataCenters": []}
        with patch.object(c, "provenance", return_value={}), patch.object(c, "budget") as budget:
            with self.assertRaises(RuntimeError):
                c.preflight(api)
            budget.assert_not_called()
        evidence = json.loads((self.out / "preflight.json").read_text())
        self.assertEqual(evidence["status"], "blocked_no_approved_gpu")
        self.assertTrue(evidence["read_only"])
        self.assertEqual(len(evidence["catalog"]), 4)
        self.assertTrue(all(x["status"] == "no_capacity" for x in evidence["catalog"]))
        self.assertTrue(all(len(call.args) == 1 for call in api.call.call_args_list))

    def test_cuda_floor_matches_catalog_evidence_and_create_for_each_gpu(self):
        expected = {
            c.MIG: "12.8",
            "NVIDIA GeForce RTX 3090": "12.4",
            "NVIDIA GeForce RTX 4090": "12.4",
            "NVIDIA GeForce RTX 5090": "12.8",
        }
        self.assertEqual(set(c.GPUS), set(expected))
        for selected, minimum in expected.items():
            with self.subTest(gpu=selected):
                api = Mock()
                api.listing.return_value = []

                def catalog(path):
                    parsed = urllib.parse.urlsplit(path)
                    name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[1])
                    query = urllib.parse.parse_qs(parsed.query)
                    self.assertEqual(query["minCudaVersion"], [expected[name]])
                    self.assertEqual(query["cloud"], ["SECURE"])
                    return {
                        "price": {"secure": 0.5},
                        "dataCenters": [{"id": "US-TEST-1", "availability": "LOW"}]
                        if name == selected
                        else [],
                    }

                api.call.side_effect = catalog
                with (
                    patch.object(c, "provenance", return_value={}),
                    patch.object(c, "budget", return_value={}),
                ):
                    evidence = c.preflight(api)
                self.assertEqual(evidence["gpu"], selected)
                self.assertEqual(evidence["min_cuda_version"], minimum)
                self.assertEqual(api.call.call_count, 4)
                self.assertEqual(
                    {x["gpu"]: x["min_cuda_version"] for x in evidence["catalog"]}, expected
                )
                evidence.update(config_sha256="b" * 64, controller_sha="a" * 40)
                api.reset_mock()
                api.call.side_effect = RuntimeError("create outcome unknown")
                with (
                    patch.object(c, "authorization_guard"),
                    patch.object(c, "preflight", return_value=evidence),
                    patch.object(c, "cleanup") as cleanup,
                    patch("builtins.print"),
                    self.assertRaises(RuntimeError),
                ):
                    c.launch(api)
                api.call.assert_called_once()
                self.assertEqual(api.call.call_args.args[:2], ("/pods", "POST"))
                payload = api.call.call_args.args[2]
                self.assertEqual(payload["gpu"]["id"], selected)
                self.assertEqual(payload["gpu"]["minCudaVersion"], minimum)
                cleanup.assert_called_once_with(api)

    def test_failed_preflight_does_not_expose_provider_error_text(self):
        api = Mock()
        api.listing.return_value = []
        api.call.side_effect = RuntimeError("Remote HTTP 403; secret-test-value")
        with patch.object(c, "provenance", return_value={}), self.assertRaises(RuntimeError):
            c.preflight(api)
        text = (self.out / "preflight.json").read_text()
        self.assertNotIn("secret-test-value", text)
        evidence = json.loads(text)
        self.assertTrue(all(x["http_status"] == 403 for x in evidence["catalog"]))
        self.assertTrue(all(x["status"] == "catalog_request_failed" for x in evidence["catalog"]))

    def test_failed_preflight_distinguishes_invalid_quote(self):
        api = Mock()
        api.listing.return_value = []
        api.call.return_value = {"price": {"secure": float("nan")}, "dataCenters": []}
        with patch.object(c, "provenance", return_value={}), self.assertRaises(RuntimeError):
            c.preflight(api)
        evidence = json.loads((self.out / "preflight.json").read_text())
        self.assertTrue(all(x["status"] == "quote_rejected" for x in evidence["catalog"]))
        self.assertTrue(all(x["rate"] is None for x in evidence["catalog"]))

    def test_budget_includes_disk_and_cleanup(self):
        for rate in (0.20, 0.69, 1.5):
            q = c.quote(rate)
            self.assertLessEqual(q["planned_total_usd"], 0.8)
            self.assertGreaterEqual(q["unallocated_reserve_usd"], 0.2)
            self.assertEqual(q["work_seconds"], 1200)
            self.assertGreater(q["disk_hourly_usd"], 0)

    def test_bad_prices_rejected(self):
        for price in (True, None, float("nan"), float("inf"), 0, -1, 1.51):
            with self.subTest(price=price), self.assertRaises(ValueError):
                c.quote(price)

    def test_cleanup_requires_repeated_absence(self):
        c.save("provision.json", self.state())
        api = Mock()
        api.listing.side_effect = [[self.pod()], [], []]
        with patch.object(c.time, "sleep"):
            c.cleanup(api)
        api.call.assert_called_once_with("/pods/pod1", "DELETE")
        self.assertEqual(api.listing.call_count, 3)
        self.assertEqual(
            json.loads((self.out / "cleanup.json").read_text())["status"], "verified_absent"
        )

    def test_cleanup_recovers_unknown_creation(self):
        state = self.state()
        state["pod_id"] = None
        c.save("provision.json", state)
        api = Mock()
        api.listing.side_effect = [[], [self.pod()], []]
        with patch.object(c.time, "sleep"):
            c.cleanup(api)
        api.call.assert_called_once_with("/pods/pod1", "DELETE")

    def test_cleanup_unknown_delete_reconciled(self):
        c.save("provision.json", self.state())
        api = Mock()
        api.listing.side_effect = [[self.pod()], [], []]
        api.call.side_effect = RuntimeError("timeout")
        with patch.object(c.time, "sleep"):
            c.cleanup(api)
        self.assertEqual(
            json.loads((self.out / "cleanup.json").read_text())["status"], "verified_absent"
        )

    def test_cleanup_refuses_unrelated_or_mounted_pods(self):
        for field in ("owner", "mount"):
            c.save("provision.json", self.state())
            pod = self.pod()
            if field == "owner":
                pod["env"][c.OWNER] = "other"
            else:
                pod["mounts"] = [{"id": "retain-me"}]
            api = Mock()
            api.listing.return_value = [pod]
            with self.assertRaises(RuntimeError):
                c.cleanup(api)
            api.call.assert_not_called()

    def test_fresh_runner_cleanup(self):
        api = Mock()
        api.listing.side_effect = [[self.pod()], [self.pod()], [], []]
        with patch.object(c.time, "sleep"):
            c.cleanup(api)
        api.call.assert_called_once_with("/pods/pod1", "DELETE")

    def test_cleanup_failure_never_claims_success(self):
        c.save("provision.json", self.state())
        api = Mock()
        api.listing.return_value = [self.pod()]
        with patch.object(c.time, "sleep"), self.assertRaises(RuntimeError):
            c.cleanup(api)
        self.assertEqual(json.loads((self.out / "cleanup.json").read_text())["status"], "unknown")

    def test_unknown_create_never_retried_and_no_account_key_in_payload(self):
        api = Mock()
        api.call.side_effect = RuntimeError("create outcome unknown")
        evidence = {
            "cost": c.quote(0.5),
            "gpu": c.MIG,
            "regions": ["US-TEST-1"],
            "config_sha256": "b" * 64,
            "controller_sha": "a" * 40,
        }
        with (
            patch.object(c, "authorization_guard"),
            patch.object(c, "preflight", return_value=evidence),
            patch.object(c, "cleanup") as cleanup,
            patch("builtins.print"),
            self.assertRaises(RuntimeError),
        ):
            c.launch(api)
        self.assertEqual(api.call.call_count, 1)
        payload = api.call.call_args.args[2]
        self.assertNotIn("RUNPOD_API_KEY", payload["env"])
        self.assertNotIn("GH_TOKEN", payload["env"])
        self.assertNotIn("FAAC_RESULT_TOKEN", (self.out / "provision.json").read_text())
        cleanup.assert_called_once_with(api)

    def test_auth_rejects_second_attempt_or_wrong_branch(self):
        for variable, value in (
            ("GITHUB_RUN_ATTEMPT", "2"),
            ("GITHUB_REF", "refs/heads/test"),
            ("CONFIRM_PAID", "false"),
            ("GITHUB_EVENT_NAME", "push"),
        ):
            with (
                patch.dict(os.environ, {variable: value}),
                patch.object(c, "git", return_value="a" * 40),
                patch.object(c.time, "time", return_value=1793664000),
                self.assertRaises(RuntimeError),
            ):
                c.authorization_guard()

    def test_authorization_ledger_rejects_previous_run(self):
        response = [
            {"object": {"sha": "a" * 40}},
            {"artifacts": [{"name": c.MARKER, "workflow_run": {"id": 99}}]},
        ]
        with (
            patch.object(c, "git", return_value="a" * 40),
            patch.object(c, "request", side_effect=response),
            patch.object(c.time, "time", return_value=1793664000),
            self.assertRaises(RuntimeError),
        ):
            c.authorization_guard()

    def test_authorization_requires_own_marker(self):
        for artifacts, allowed in (
            ([], False),
            ([{"name": c.MARKER, "workflow_run": {"id": 1234}}], True),
        ):
            response = [{"object": {"sha": "a" * 40}}, {"artifacts": artifacts}]
            with (
                patch.object(c, "git", return_value="a" * 40),
                patch.object(c, "request", side_effect=response),
                patch.object(c.time, "time", return_value=1793664000),
            ):
                if allowed:
                    c.authorization_guard()
                else:
                    with self.assertRaises(RuntimeError):
                        c.authorization_guard()

    def test_download_uses_same_headers_and_checks_bytes(self):
        data = b"diagnostic archive"
        expected = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        opener = Mock()
        opener.open.return_value = io.BytesIO(data)
        with patch.object(c.urllib.request, "build_opener", return_value=opener):
            c.download("https://test.invalid/archive", "test-token", expected)
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_header("User-agent"), "flashattentionanycard-actions/0.1")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-token")
        self.assertEqual((self.out / "smoke.tar.gz").read_bytes(), data)

    def test_inline_diagnostics_validated_without_network(self):
        data = b"diagnostic archive"
        expected = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with patch.object(c.urllib.request, "build_opener") as network:
            c.download(
                "https://test.invalid/archive",
                "test-token",
                expected,
                base64.b64encode(data).decode(),
            )
            network.assert_not_called()
            with self.assertRaises(RuntimeError):
                c.download(
                    "https://test.invalid/archive",
                    "test-token",
                    expected,
                    base64.b64encode(b"corrupt").decode(),
                )
            with self.assertRaises(RuntimeError):
                c.download("https://test.invalid/archive", "test-token", expected, "a" * 180001)

    def test_reconciled_retry_preserves_total_cap(self):
        self.assertEqual(c.PRIOR_SPEND_RESERVE, 0.06)
        self.assertGreater(c.PRIOR_SPEND_RESERVE, 0.0483414)
        for rate in (0.5, 0.69, 1.5):
            q = c.quote(rate)
            estimate = (
                c.PRIOR_SPEND_RESERVE
                + (rate + c.DISK_HOURLY) * (q["work_seconds"] + q["cleanup_seconds"]) / 3600
            )
            self.assertAlmostEqual(q["planned_total_usd"], estimate)
            self.assertLessEqual(estimate, 0.8)

    def test_only_exact_reconciled_marker_allows_retry(self):
        for marker_id, allowed in ((13204770313, True), (999, False), (None, False)):
            previous = {"name": c.MARKER, "workflow_run": {"id": 41577245210}, "id": marker_id}
            current = {"name": c.MARKER, "workflow_run": {"id": 1234}, "id": 888}
            response = [{"object": {"sha": "a" * 40}}, {"artifacts": [previous, current]}]
            with (
                patch.object(c, "git", return_value="a" * 40),
                patch.object(c, "request", side_effect=response),
                patch.object(c.time, "time", return_value=1793664000),
            ):
                if allowed:
                    c.authorization_guard()
                else:
                    with self.assertRaises(RuntimeError):
                        c.authorization_guard()

    def test_zero_create_marker_reconciliation_does_not_admit_unknown_attempt(self):
        current = {"name": c.MARKER, "workflow_run": {"id": 1234}, "id": 888}
        first = {"name": c.MARKER, "workflow_run": {"id": 41577245210}, "id": 13204770313}
        for run, marker, allowed in (
            (41668287039, 13245909643, True),
            (41668287039, 999, False),
            (41668287039, None, False),
            (999, 13245909643, False),
        ):
            with self.subTest(run=run, marker=marker):
                previous = {"name": c.MARKER, "workflow_run": {"id": run}, "id": marker}
                response = [
                    {"object": {"sha": "a" * 40}},
                    {"artifacts": [first, previous, current]},
                ]
                with (
                    patch.object(c, "git", return_value="a" * 40),
                    patch.object(c, "request", side_effect=response),
                    patch.object(c.time, "time", return_value=1793664000),
                ):
                    if allowed:
                        c.authorization_guard()
                    else:
                        with self.assertRaises(RuntimeError):
                            c.authorization_guard()

    def test_pagination_must_be_complete(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "test"}):
            api = Runpod()
        with patch.object(api, "call", return_value={"pods": []}), self.assertRaises(RuntimeError):
            api.listing("/pods", "pods")

    def test_pinned_workload_matches_current_checkout(self):
        self.assertEqual(c.provenance()["image_source_sha"], SOURCE)

    def test_archive_rejects_traversal_symlink_and_duplicate(self):
        for kind in ("traversal", "symlink", "duplicate", "checksum"):
            path = self.out / "bad.tar.gz"
            with tarfile.open(path, "w:gz") as tar:
                info = tarfile.TarInfo(
                    "../outside" if kind == "traversal" else "faac-smoke/checksums.json"
                )
                data = b"{}"
                info.size = len(data)
                if kind == "symlink":
                    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                    info.size = 0
                tar.addfile(info, io.BytesIO(data))
                if kind == "duplicate":
                    tar.addfile(info, io.BytesIO(data))
            with self.assertRaises(RuntimeError):
                verify_archive(path)


if __name__ == "__main__":
    unittest.main()
