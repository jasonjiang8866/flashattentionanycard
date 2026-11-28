# Bounded single-GPU large-shape pilot

Implemented under the user's **2026-11-28 US$10 / 500-minute** authorization.
Status: controller and CPU acceptance tests implemented; **GPU pilot not run**.
M1's accepted small-shape smoke remains documented in [M1_ACCEPTANCE.md](M1_ACCEPTANCE.md).

## Fixed workload and image

| Item | Pilot contract |
|---|---|
| Workloads | FP16 GEMM 2048³ and RMSNorm 4096×4096 |
| Search | Grid / Random / multivariate TPE; seeds 0 and 1; full categorical domains |
| Budgets | GEMM 24 attempts per method/seed; RMSNorm 12; failures/pruning/duplicates consume attempts |
| Total | 12 studies, exactly 216 attempts, 12 frozen winners × 6 pairs = 72 paired workers |
| Correctness | At least 25 GPU tests; no skipped, failed or errored tests |
| Comparison | Fixed PyTorch baselines, held-out input seeds, alternating order, raw samples and independently recomputed reports |
| Source | `f979f1af25b5e9e4a3254ec2e195cf681c61e84d` |
| Image | `ghcr.io/jasonjiang8866/flashattentionanycard-runpod@sha256:972e5e3fbfbc80fde96cbfadfd117e5501fa81d6ce4899295dc8122af632f37d` |

The existing image contains the pilot runner and M1 confirmation implementation.
Only the injected worker, Actions controller and CPU verifier change. Admission
checks that current workload code/configuration still matches the image's source.
No arbitrary suite file, warm-start source, seed or repetition override is exposed
by the paid workflow. This slice covers one cold single-GPU pilot, not transfer.

## Budget and selection

- Total USD cap: **$10**, conservatively including the prior **$0.41** reserved
  spend. This is not an additional $10 on top of the old allowance.
- Total pilot time allowance: **500 minutes**, including failed attempts and cleanup.
  First attempt: at most **120 minutes work + 10 minutes cleanup**. Work includes
  allocation/image startup, tests, search, confirmation and CPU verification;
  the worker stops 180 seconds before the work deadline to allow result transfer.
- Keep at least **$1 unallocated**. Compute and 40 GB ephemeral disk are included;
  transfer is estimated at $0 under the existing pricing model. At the unchanged
  $1.50/hour GPU ceiling, the planned cumulative estimate is **$3.672897** over
  130 minutes including cleanup. This is a ceiling-rate illustration, not a live quote.
- Quote admission shortens work to fit both remaining money and remaining time;
  less than 30 useful work minutes is rejected. It also checks available account
  credit and the $500 monthly guard, projecting existing account spend over the
  full planned window. These checks are estimates, not provider-enforced caps.
- Prefer an available, within-cap **RTX 4090**; otherwise choose the cheapest
  allowed available 3090 / 5090 / PRO 6000 Blackwell MIG 24GB. Record the complete
  catalog and selection reason. No automatic second create after a capacity error
  or unknown response. Host CUDA uses lower bounds (3090/4090 ≥12.4; Blackwell ≥12.8),
  while the fixed container runtime remains CUDA 12.8; GPU tests decide compatibility.

Pilot authorization ID: `faac-pilot-20261128-usd10-500m`.
Conservative launch cutoff: **2026-12-05 00:00 UTC / 08:00 SGT**, before the retained
marker's 90-day expiry. The old smoke authorization stays expired. A marker is uploaded
before the single create request; any unreconciled previous pilot marker blocks another
run, including re-dispatch. `run_attempt > 1` is rejected. Before a retry, review its
exact run/marker, cleanup, conservative spend and elapsed time, then update the ledger
without resetting either allowance. There are no retries in this implementation.

The work alarm starts before POST /pods and triggers cleanup on timeout, SIGTERM or
SIGINT. Cleanup matches repository, authorization, run name/ID and Pod ID; it refuses
unrelated resources and persistent mounts. A workflow `always()` step and a separate
`workflow_run` recovery job verify absence. No persistent volume is created or deleted.
Deletion failures can exceed estimates and must be reported as unverified cleanup.

## Review and dispatch

The new workflow must be merged before GitHub exposes its manual entrypoint on main.
Use [RunPod single-GPU pilot](https://github.com/jasonjiang8866/flashattentionanycard/actions/workflows/runpod-gpu-pilot.yml).
Code merge and GPU execution are distinct steps; the user's US$10 / 500-minute grant
is already recorded above and must not be requested again for this scope/window.

1. Re-read current main and passing CI; pass its full SHA as `expected_sha`.
2. Run with `confirm_paid_pilot=false` for live read-only inventory, price, account
   guard and image/source checks. It neither records a paid marker nor creates a Pod.
3. Inspect that evidence, then dispatch once with `confirm_paid_pilot=true` and the
   same reviewed SHA. Paid admission repeats live preflight; it never relies on an old quote.
4. Follow progress from allocation/image startup through GPU correctness, search,
   confirmation, report transfer/CPU verification and cleanup. Only allowlisted
   counters and phases appear in progress; account credentials stay in Actions Secrets.
5. Download the result artifact and independently verify it, then confirm the
   originating run's cleanup recovery succeeded. A green workflow alone is insufficient.

CPU-only plan (no credentials, inventory or compute; default rate is the ceiling):

```bash
python -m runpod.paid_smoke plan --suite pilot
python -m runpod.launch pilot --confirm-repetitions 6 --dry-run
```

After retrieving the artifact in a CPU environment with Optuna 4.9.0:

```bash
python -m runpod.verify_paid /path/to/pilot.tar.gz --suite pilot
```

The verifier accepts only the declared `faac-pilot` archive root and fixed pilot
plan; rejects traversal, links, duplicate/aliased paths and oversized expansion;
checks every member hash and study receipt; requires the exact search set,
method/seed/workload/protocol identity and full 24/12 attempt sequences; verifies
all 72 held-out paired workers, source-result bindings and sample-derived CSV
statistics. During Actions acceptance, archive hardware must also match the
actually allocated GPU. Smoke archives cannot satisfy pilot acceptance.

## Completion evidence and limits

Require all of: `acceptance.json` reports `accepted`, 12 studies, 216 attempts,
12 confirmations and 72 paired workers; the downloaded archive independently
verifies; original and recovery cleanup confirm absence. Retain allocation,
provision intent/deadline, timing, preflight, artifact hashes and both cleanup
records. Charge the conservative intent-to-independent-absence interval to the
same budget before another attempt. A timeout retains partial diagnostics and
does not count as a completed fair-budget pilot.

Two search seeds on one GPU support exploratory comparisons only. Report failures,
regressions and noise; do not infer cross-GPU transfer or general TPE superiority.
Full details of the search protocol are in [PILOT.md](PILOT.md).
