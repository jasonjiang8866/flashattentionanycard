# One-off $1 RunPod GPU smoke

This entrypoint uses `RUNPOD_API_KEY` only in GitHub Actions and reuses the verified
[2026-10-30 image publication](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/41028932034).
It does not publish an image. Existing [pilot/bootstrap instructions](PILOT.md) remain available.

## Launch specification

- Workflow: [RunPod GPU smoke ($1 authorization)](https://github.com/jasonjiang8866/flashattentionanycard/actions/workflows/runpod-gpu-smoke.yml).
- Ref: `main`. Input `expected_sha`: full current main commit SHA. Input
  `confirm_paid_smoke`: `true` for the authorized experiment; `false` runs read-only preflight.
- Authorization: `faac-20261103-usd1`, one creation attempt, $1 total including cleanup and
  temporary disk. Expires 2026-11-10 UTC. No automatic retry or second GPU. All four GPU
  types were authorized: RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, RTX 3090,
  RTX 4090, RTX 5090. Select the cheapest verified available Secure Cloud GPU, one card,
  maximum $1.50/hour, CUDA >=12.8. No capacity or unverifiable price means no creation.
- Workload: existing `smoke` suite, 18 studies / at most 36 candidate attempts;
  GEMM, RMSNorm, grouped MoE GEMM, FP16/BF16; Grid/Random/TPE; sampler seed 0,
  two validation seeds, synthetic inputs identified by the config hash in artifacts.
  This is a CUDA functionality/measurement smoke, not the two-GPU pilot or a
  cross-architecture warm-start performance result.
- Image: `ghcr.io/jasonjiang8866/flashattentionanycard-runpod@sha256:079df18c23caee6769ee0b70570c5c12863d7e5d0f847e7fb7cec1fb4d1136c3`.
  Workload source: `14118cbec5aba88654232b1a2f232be4ca35d1db`. The controller is the
  workflow checkout SHA. Preflight requires all workload/runner/config/dependency files
  to match the image source; injected controller worker code is hashed separately.
- Runtime: at most 20 minutes including provisioning, image pull, CUDA tests, search,
  analysis and archive retrieval; stop work 90 seconds early for result transfer.
  Reserve another 6 minutes for cleanup. At the maximum GPU rate the compute+disk
  estimate is below $0.66, leaving more than $0.34 of the $1 authorization unused.
- Storage: 40 GB ephemeral container disk, no network/global/persistent volume created.
  Current [official pricing](https://docs.runpod.io/pods/pricing) is $0.10/GB/month,
  charged per second; controller conservatively uses a 28-day month (~$0.00596/hour).
  RunPod ingress/egress is free under that pricing. No account top-up is performed.
  The account check requires sufficient credits, including the one-hour deployment
  minimum, and reported UTC monthly charges plus reserve below $500.

The budget is enforced by conservative rate/runtime admission and cleanup, **not a
provider-enforced dollar cap**. Provider outages or failed deletion can exceed the estimate;
inspect any cleanup failure immediately. Account billing can lag. This workflow does not
change account autopay or inspect unrelated credentials.

## Execution and evidence

A retained Actions authorization artifact is uploaded **before** the sole `POST /pods`.
The controller refuses another run with the same authorization, `run_attempt > 1`, a
non-manual event, a branch other than main, moved main, or an expired authorization.
Do not delete the authorization artifact to retry: a failed/unknown attempt requires
inventory and spending reconciliation before any newly authorized experiment.

The random result-only token is masked, is never written to artifacts, and cannot call
RunPod. The account key and GitHub token never enter the Pod. The worker serves a
bounded result/archive over an authenticated HTTPS proxy. Startup never clones code
or installs packages inside the paid Pod.

Acceptance requires all of the following:

1. At least 19 GPU correctness tests, zero failures/errors/skips.
2. All 18 smoke studies complete with valid timing samples, numerical validation,
   attempt accounting, best trial consistency and per-study receipts.
3. Exact source/config/software fingerprints, reports and all archive member hashes.
   The CPU Actions runner independently validates the downloaded archive, including
   traversal/symlink/duplicate rejection and bounded decompression.
4. `cleanup.json` confirms the exact owned temporary Pod absent from independent
   provider inventories. `acceptance.json` alone is not full experiment acceptance.

Artifacts include `preflight.json`, `provision.json`, `allocation.json`, `result.json`,
`smoke.tar.gz`, `acceptance.json` and `cleanup.json` when the corresponding stages
complete. Partial results remain partial; a CPU test pass is not CUDA execution.
Costs in cleanup evidence are conservative estimates, not final account invoices.

## Cleanup and cancellation

The controller reconciles unknown creation by exact name/run ID and never retries
creation. `finally` performs deletion; an `always()` step independently checks again.
A separate [cleanup recovery workflow](https://github.com/jasonjiang8866/flashattentionanycard/actions/workflows/runpod-gpu-cleanup.yml)
runs after the manual workflow completes, including failure/cancellation, and checks
only that originating run's ownership tags. It can recover identity/timing from the
owned Pod if the original runner was lost. This recovery workflow never creates compute.
It is a fallback, not proof that deletion happened immediately.

Do not just stop/cancel Actions and assume billing ended. Verify cleanup artifact
`status=verified_absent`; unknown or missing cleanup evidence requires inspection of
`faac-20261103-usd1-<Actions run ID>`. Unexpected ownership or persistent mounts are
not deleted automatically. No persistent volume is deleted by these controllers.

## Validation before merge

Run `python -m unittest discover -s runpod/tests -v`, repository CPU pytest, Ruff and
workflow YAML validation. These checks use mocks and local synthetic files; they
make no paid requests. Live API/schema, availability, account budget and actual CUDA
results remain unverified until the manual workflow runs.
