"""CPU-only lifecycle/budget tests. No provider calls or CUDA measurements."""

import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from runpod import paid_smoke as c
from runpod import paid_worker, provider
from runpod.paid_policy import PILOT
from runpod.tests import test_m1_verification as fixtures
from runpod.tests.test_m1_verification import sha, write

GPU = "NVIDIA GeForce RTX 4090"


class PilotArchiveTests(fixtures.ArchiveTests):
    suite = "pilot"

    def test_wrong_suite_gpu_or_plan_rejected(self):
        archive = self.archive()
        with self.assertRaises(RuntimeError):
            from runpod.verify_paid import verify_archive

            verify_archive(archive)
        with self.assertRaisesRegex(RuntimeError, "allocated GPU"):
            self.verify(archive, expected_gpu=GPU)
        path = self.output / "plan.json"
        plan = copy.deepcopy(self.plan)
        plan[0]["seed"] = 42
        write(path, plan)
        with self.assertRaisesRegex(RuntimeError, "study plan"):
            self.verify(self.archive())

    def test_rehashed_short_search_cannot_claim_full_pilot(self):
        folder = self.output / self.plan[0]["path"]
        result = json.loads((folder / "result.json").read_text())
        result["trials"].pop()
        result["attempts"] -= 1
        result["successful_evaluations"] -= 1
        write(folder / "result.json", result)
        write(
            folder / "receipt.json",
            {name: sha(folder / name) for name in ("result.json", "trials.csv", "study.sqlite3")},
        )
        with self.assertRaisesRegex(RuntimeError, "equal attempt budget"):
            self.verify(self.archive())

    def test_extra_search_even_with_valid_outer_hashes_is_rejected(self):
        write(self.output / "cases/extra/grid-seed-0/result.json", {})
        with self.assertRaisesRegex(RuntimeError, "search study set"):
            self.verify(self.archive())


class PilotControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        for p in (
            patch.object(c, "OUT", self.out),
            patch.dict(
                c.os.environ,
                {
                    "GITHUB_REPOSITORY": provider.REPOSITORY,
                    "GITHUB_RUN_ID": "1234",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_EVENT_NAME": "workflow_dispatch",
                    "GITHUB_REF": "refs/heads/main",
                    "CONFIRM_PAID": "true",
                    "EXPECTED_SHA": "a" * 40,
                    "GH_TOKEN": "test-only",
                },
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_budget_caps_money_time_storage_cleanup_and_carries_spend(self):
        for rate in (0.5, 0.69, 0.89, 1.39, 1.5):
            q = c.quote(rate, suite="pilot")
            self.assertEqual(q["work_seconds"], 7200)
            self.assertEqual(q["cleanup_seconds"], 600)
            self.assertEqual(q["transfer_seconds"], 180)
            self.assertLessEqual(q["planned_total_seconds"], 500 * 60)
            self.assertLessEqual(q["planned_total_usd"], 9)
            self.assertAlmostEqual(
                q["planned_total_usd"], 0.41 + (rate + c.DISK_HOURLY) * 7800 / 3600
            )
        for rate in (float("nan"), float("inf"), 0, -1, True, 1.51):
            with self.assertRaises(ValueError):
                c.quote(rate, suite="pilot")
        # Future reconciliations cannot spend time or dollars twice.
        with patch.object(c, "PILOT", replace(PILOT, prior_seconds=450 * 60)):
            self.assertEqual(c.quote(1.5, suite="pilot")["work_seconds"], 40 * 60)
        for limits in (replace(PILOT, prior_usd=8.9), replace(PILOT, prior_seconds=490 * 60)):
            with patch.object(c, "PILOT", limits), self.assertRaises(ValueError):
                c.quote(1.5, suite="pilot")

    def test_plan_cli_needs_neither_credentials_nor_network(self):
        with (
            patch.dict(c.os.environ, {}, clear=True),
            patch.object(c, "Runpod") as api,
            patch("sys.argv", ["paid_smoke", "plan", "--suite", "pilot"]),
            redirect_stdout(io.StringIO()) as output,
        ):
            c.main()
        api.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(
            (plan["studies"], plan["search_attempts"], plan["paired_workers"]), (12, 216, 72)
        )
        self.assertEqual(list(self.out.iterdir()), [])

    def test_pilot_marker_expiry_and_smoke_authorization_are_separate(self):
        own = {"name": PILOT.marker, "id": 900, "workflow_run": {"id": 1234}}
        previous = {"name": PILOT.marker, "id": 901, "workflow_run": {"id": 5678}}
        for markers, offset, allowed in (
            ([own], -1, True),
            ([own, previous], -1, False),
            ([], -1, False),
            ([own], 0, False),
        ):
            with (
                patch.object(c, "git", return_value="a" * 40),
                patch.object(c.time, "time", return_value=PILOT.expires + offset),
                patch.object(
                    c,
                    "request",
                    side_effect=[{"object": {"sha": "a" * 40}}, {"artifacts": markers}],
                ),
            ):
                if allowed:
                    c.authorization_guard(suite="pilot")
                else:
                    with self.assertRaises(RuntimeError):
                        c.authorization_guard(suite="pilot")
        with (
            patch.object(c, "git", return_value="a" * 40),
            patch.object(c.time, "time", return_value=PILOT.expires - 1),
            patch.object(c, "request") as network,
            self.assertRaisesRegex(RuntimeError, "expiry"),
        ):
            c.authorization_guard(suite="smoke")
        network.assert_not_called()

    def test_4090_preferred_with_explicit_capacity_fallback(self):
        api = Mock()
        api.listing.return_value = []

        def catalog(path):
            is_4090 = "4090" in path
            return {
                "price": {"secure": 0.89 if is_4090 else 0.69},
                "dataCenters": [{"id": "EU-TEST-1", "availability": "HIGH"}]
                if not (is_4090 and unavailable)
                else [],
            }

        api.call.side_effect = catalog
        for unavailable in (False, True):
            with (
                patch.object(c, "provenance", return_value={}),
                patch.object(c, "budget", return_value={}),
            ):
                result = c.preflight(api, suite="pilot")
            self.assertEqual(result["gpu"] == GPU, not unavailable)
            self.assertEqual(result["plan"]["paired_workers"], 72)
            self.assertTrue(
                all("minCudaVersion=" in call.args[0] for call in api.call.call_args_list)
            )
        api.listing.return_value = [{"env": {c.OWNER: provider.REPOSITORY}}]
        with (
            patch.object(c, "provenance", return_value={}),
            self.assertRaisesRegex(RuntimeError, "owned Pod"),
        ):
            c.preflight(api, suite="pilot")

    def test_pilot_unknown_create_never_retries_and_always_cleans_up(self):
        api = Mock()
        api.call.side_effect = RuntimeError("unknown create outcome")
        evidence = {
            "cost": c.quote(0.89, suite="pilot"),
            "gpu": GPU,
            "regions": ["EU-TEST-1"],
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
            c.launch(api, suite="pilot")
        self.assertEqual(api.call.call_count, 1)
        body = api.call.call_args.args[2]
        self.assertEqual(body["env"]["FAAC_SUITE"], "pilot")
        self.assertEqual(body["env"]["FAAC_AUTHORIZATION"], PILOT.authorization)
        self.assertNotIn("RUNPOD_API_KEY", body["env"])
        state = json.loads((self.out / "provision.json").read_text())
        self.assertAlmostEqual(state["deadline_epoch"] - float(body["env"]["FAAC_DEADLINE"]), 180)
        cleanup.assert_called_once_with(api, suite="pilot")

    def test_cleanup_cli_routes_to_pilot_and_never_deletes_foreign_state(self):
        api = Mock()
        api.listing.return_value = []
        with (
            patch("sys.argv", ["paid_smoke", "cleanup", "--suite", "pilot"]),
            patch.object(c, "Runpod", return_value=api),
            patch.object(c.time, "sleep"),
        ):
            c.main()
        self.assertEqual(
            json.loads((self.out / "cleanup.json").read_text())["status"], "verified_absent"
        )
        c.save("provision.json", {"name": c.AUTHORIZATION + "-1234", "run_id": "1234"})
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            c.cleanup(api, suite="pilot")
        self.assertEqual(json.loads((self.out / "cleanup.json").read_text())["status"], "unknown")
        api.call.assert_not_called()

    def test_worker_selects_pilot_and_redacts_progress(self):
        with patch.object(paid_worker, "SUITE", "pilot"):
            self.assertEqual(paid_worker.suite_command()[3], "pilot")
        write(
            self.out / "status.json",
            {
                "phase": "confirmations/cases/gemm_fp16/tpe-seed-1",
                "completed_studies": 12,
                "completed_confirmations": 3,
                "secret": "do-not-export",
            },
        )
        with (
            patch.object(paid_worker, "OUTPUT", self.out),
            patch.object(paid_worker, "RESULT", {"status": "running"}),
        ):
            result = paid_worker.progress()
        self.assertEqual(result["completed_confirmations"], 3)
        self.assertNotIn("secret", result)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Archive verifier requires Optuna")
    def test_mock_provider_download_real_archive_verifier_and_real_cleanup(self):
        # Synthetic archive + mocked transport; exercise the real controller,
        # downloader, semantic verifier and cleanup together without a GPU/API.
        fixture = PilotArchiveTests()
        fixture.hardware_name = GPU
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        data = fixture.archive().read_bytes()
        evidence = {
            "cost": c.quote(0.89, suite="pilot"),
            "gpu": GPU,
            "regions": ["EU-TEST-1"],
            "config_sha256": "b" * 64,
            "controller_sha": "a" * 40,
        }
        result = {
            "status": "accepted",
            "source_sha": provider.SOURCE,
            "config_sha256": "b" * 64,
            "controller_sha": "a" * 40,
            "studies": 12,
            "confirmations": 12,
            "paired_workers": 72,
            "gpu_tests": 25,
            "archive": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
        }
        api = Mock()
        pod = {}

        def call(path, method="GET", body=None):
            if method == "POST":
                pod.update(body, id="pilot123", dataCenterId="EU-TEST-1", cost=0.89)
                return {"id": "pilot123"}
            if method == "DELETE":
                return None
            return pod

        api.call.side_effect = call
        api.listing.side_effect = [[pod], [], [], []]
        opener = Mock()
        opener.open.return_value = io.BytesIO(data)
        with (
            patch.object(c, "authorization_guard"),
            patch.object(c, "preflight", return_value=evidence),
            patch.object(
                c,
                "request",
                side_effect=[{"status": "running", "phase": "gpu_correctness"}, result],
            ),
            patch.object(c.urllib.request, "build_opener", return_value=opener),
            patch.object(c.time, "sleep"),
            patch("builtins.print"),
        ):
            c.launch(api, suite="pilot")
        accepted = json.loads((self.out / "acceptance.json").read_text())
        self.assertEqual(
            (accepted["studies"], accepted["search_attempts"], accepted["paired_workers"]),
            (12, 216, 72),
        )
        self.assertEqual((self.out / "pilot.tar.gz").read_bytes(), data)
        cleanup = json.loads((self.out / "cleanup.json").read_text())
        self.assertEqual(cleanup["status"], "verified_absent")
        self.assertTrue(cleanup["within_budget"] and cleanup["within_time"])
        self.assertEqual(sum(call.args[1:2] == ("POST",) for call in api.call.call_args_list), 1)
        self.assertEqual(sum(call.args[1:2] == ("DELETE",) for call in api.call.call_args_list), 1)

    def test_deadline_and_cancellation_always_enter_pilot_cleanup(self):
        evidence = {
            "cost": c.quote(0.89, suite="pilot"),
            "gpu": GPU,
            "regions": ["EU-TEST-1"],
            "config_sha256": "b" * 64,
            "controller_sha": "a" * 40,
        }
        for signum in (c.signal.SIGALRM, c.signal.SIGTERM, c.signal.SIGINT):
            api = Mock()

            def interrupt_create(*args):
                c.interrupt(signum, None)

            api.call.side_effect = interrupt_create
            with (
                patch.object(c, "authorization_guard"),
                patch.object(c, "preflight", return_value=evidence),
                patch.object(c, "cleanup") as cleanup,
                patch("builtins.print"),
                self.assertRaises(c.Deadline),
            ):
                c.launch(api, suite="pilot")
            self.assertEqual(api.call.call_count, 1)
            cleanup.assert_called_once_with(api, suite="pilot")

    def test_monthly_guard_accounts_for_other_pods_during_full_horizon(self):
        api = Mock(key="test-only")
        response = {
            "data": {
                "myself": {
                    "clientBalance": 20,
                    "currentSpendPerHr": 2,
                    "dailyCharges": [{"amount": 487}],
                }
            }
        }
        with patch.object(provider, "request", return_value=response):
            provider.budget(api, reserve=9.59, horizon_hours=1)
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                provider.budget(api, reserve=9.59, horizon_hours=7800 / 3600)


if __name__ == "__main__":
    unittest.main()
