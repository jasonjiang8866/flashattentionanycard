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
  maximum $1.50/hour. Host CUDA floors are >=12.4 for RTX 3090/4090 and >=12.8
  for RTX 5090/PRO 6000 Blackwell MIG. No capacity or unverifiable price means no creation.
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
  estimate plus the $0.06 prior-spend reserve is below $0.72, leaving more than
  $0.28 of the same $1 authorization unused.
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

### Host CUDA admission and runtime compatibility

| GPU | Minimum host-reported CUDA version |
| --- | --- |
| RTX 3090 (Ampere) | 12.4 |
| RTX 4090 (Ada) | 12.4 |
| RTX 5090 (Blackwell) | 12.8 |
| RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | 12.8 |

Catalog queries and Pod creation use the same per-GPU minimum; each catalog record
and selected offer in `preflight.json` retain it. The minimum is a host-selection
filter, not a change to the pinned image's torch 2.9.0+cu128 / Triton 3.5.0 runtime.
Newer host versions remain eligible. Secure Cloud, approved GPU types and budget
limits are unchanged.

CUDA 12.4 is a conservative experimental floor for Ampere/Ada, not a claim that
every cu128 workload works on every such host. NVIDIA documents CUDA 12.x minor
version compatibility with a sufficiently new driver, but newer driver features
and driver-side PTX JIT can still fail. The existing CUDA probe, all GPU correctness
tests and study validation must pass on the actual host; failures reject the run
and trigger cleanup. Blackwell retains the 12.8 floor. No driver replacement,
compatibility-library injection or test bypass is used.

References: [NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/610/minor-version-compatibility.html),
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/),
[NVIDIA architecture support matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).

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

## Reconciled first attempt and bounded retry

[Run 41577245210](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/41577245210)
selected an RTX 3090 at $0.50/hour after capacity returned. The suite failed after
6.73 seconds; binary archive retrieval then returned HTTP 403. This is **rejected**,
not an accepted CUDA result. The exact suite error was not available in the first
worker's summary. Pod `z69aqds1wz4d90` was deleted and its absence independently
confirmed by [recovery 41577845509](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/41577845509).
The conservative elapsed-time estimate, including disk, was $0.0483414.

The same user authorization explicitly includes retries. Reserve **$0.06** for this
reconciled attempt and admit only one further attempt under the **same $1 aggregate
cap**. Keep the original authorization artifact; only its exact run/artifact identity
is reconciled. Any additional unreconciled run still blocks launch. New cost records
include the prior spend reserve. No automatic retry was added.

Binary archive requests now use the same explicit headers as JSON result requests.
Small archives (<=128 KiB) are also carried in the authenticated JSON result and must
pass the same exact size/hash checks. Suite phase/error evidence is retained in the
summary even when a separate download fails. The cause of the original HTTP 403
and suite failure remains unconfirmed until live diagnostic evidence is recovered.

### Reconciled zero-create attempt

[Run 41668287039](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/41668287039)
uploaded its authorization marker, then failed the second read-only preflight with
`no_capacity` for all four GPUs under the original 12.8 floor. It failed before
`provision.json` and before `POST /pods`; no paid resource was requested. Its result
artifact contains only `preflight.json` and `cleanup.json`, with SHA256
`bec60024869e410a80625c8ffa3a9e51d7afa235a1c8ee1b6bcfe68d94ce9359`.
[Independent recovery 41668355419](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/41668355419)
also verified the exact run's Pod absent. Additional RunPod cost: $0.

Preserve marker artifact `13245909643` and reconcile only that exact run/artifact
pair alongside the first attempt. The prior-spend reserve remains $0.06, the total
authorization remains $1, and any other unreconciled attempt still blocks launch.
The changed CUDA floor has not yet been validated on a real GPU.
