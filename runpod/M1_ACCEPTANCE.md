# M1 GPU acceptance handoff

M1 implementation was merged in [PR #11](https://github.com/jasonjiang8866/flashattentionanycard/pull/11)
at `f979f1af25b5e9e4a3254ec2e195cf681c61e84d`. The image publication is
[run 43898896512](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43898896512).
The controller's `IMAGE` and `SOURCE` pins reference that verified publication.
Its immutable digest is `sha256:972e5e3fbfbc80fde96cbfadfd117e5501fa81d6ce4899295dc8122af632f37d`.
Publication artifact `14241368867` has ZIP SHA-256
`35438684b94adb3bc1a305e4a868da3e072ba0ea80a7ce6a7e8105e7e9e3dcb8`.
The published image passed 156 offline tests, with 25 GPU tests deselected and zero skipped.
This handoff does not itself prove real CUDA execution or performance improvement.

## Bounded workload

- Same synthetic `smoke` inputs: three kernels, FP16/BF16, Grid/Random/TPE,
  18 studies and at most 36 search attempts.
- `--confirm-repetitions 6`: freeze each winner and run 108 fresh paired workers
  in total, with held-out input seeds and alternating candidate/PyTorch order.
- At least 25 GPU tests must execute with zero skips, failures or errors.
- All 18 confirmation receipts, raw pairs and copied source results must verify.
  The CPU Actions runner independently checks that each confirmation belongs to
  its corresponding accepted study and that `baseline-summary.csv` matches the
  recomputed summaries exactly. Missing, partial, substituted or extra results
  reject acceptance even if the archive's outer checksums are valid.
- Timing/precision and statistical limitations remain those in
  [MEASUREMENT.md](MEASUREMENT.md). This small functional smoke is not the full
  cross-GPU performance pilot and does not promise speedups over PyTorch.

The injected worker enables M1 confirmations explicitly. Search-only launch defaults
remain unchanged. A newly published workload image contains the M1 kernels, tests
and confirmation code; controller-only changes are injected or run on Actions.
PR image validation reads the existing build cache without exporting another copy;
publication continues to refresh the cache. This avoids the observed PR cache-upload
stall while preserving image builds and all offline tests.
The paid Actions job installs `optuna==4.9.0` for CPU artifact verification before
any resource admission. No dependencies are installed inside a paid Pod.

## Budget and previous-run reconciliation

The user's project allowance is $10 / 500 minutes. This one attempt retains the
stricter existing **$1 cumulative controller cap**, including earlier reserved spend,
temporary disk, retries already reconciled and cleanup. It neither resets the
ledger nor allocates the entire project allowance.

M0 run `43743010737` consumed marker `14173177303`; retain it and reconcile only
that exact pair. Re-downloaded original/recovery ZIP hashes matched the records in
[GPU_SMOKE.md](GPU_SMOKE.md). Both cleanup records confirmed absence three times.
Create intent `1794764824.8761477` to independent absence `1794765143.290164`
is 318.414016 seconds. At the recorded $0.69/hour plus temporary disk rate this
reserves $0.061555832, bringing the earlier $0.16 reserve to $0.221555832.
Round the cumulative reserve up to **$0.23**; this is an estimate, not an invoice.
Any other unreconciled attempt blocks creation. No marker is deleted.

The single-Pod work window remains at most 1,200 seconds, with 360 seconds reserved
for cleanup and 90 seconds of the work window reserved for result transfer. The
controller shortens the work window when necessary to keep the planned cumulative
cost at or below $0.80, leaving at least $0.20 unallocated under its $1 cap.
At the maximum $1.50/hour GPU quote the work window is 1,002 seconds; at $0.69/hour
it is 1,200 seconds. These are admission estimates, not provider-enforced caps.
Fresh API quotes/capacity, monthly guard and exact image/source equality still apply.
The GPU allowlist, CUDA minimums, $1.50/hour ceiling, 40 GB ephemeral disk,
one-create/no-automatic-retry policy and independent cleanup are unchanged.

## Dispatch and completion

After the handoff is reviewed and merged, re-read current `main` and run
`runpod-gpu-smoke.yml` with that full `expected_sha`. A read-only preflight uses
`confirm_paid_smoke=false`; the authorized single attempt uses `true` only after
reviewing that evidence. The existing authorization cutoff remains 2026-11-23
00:00 UTC (08:00 SGT), before all retained marker expiries.

Completion requires all of: `acceptance.json` reports 18 confirmations and 108
paired workers, the retrieved artifact independently verifies, and the owned Pod
is confirmed absent by the cleanup and recovery workflows. A green image build
or a successful search-only archive is insufficient. Record measured results and
cost/cleanup evidence only after that run actually finishes.
