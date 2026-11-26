# M1 GPU acceptance record

**Accepted on 2026-11-17 SGT (2026-11-16 UTC); evidence reverified on 2026-11-27 SGT.**
[Run 43917307722](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43917307722)
completed the M1 smoke and
[recovery 43920030853](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43920030853)
independently confirmed the owned temporary Pod absent. Both workflows succeeded.
This is single-device functional acceptance, not completion of the cross-GPU pilot.

M1 implementation was merged in [PR #11](https://github.com/jasonjiang8866/flashattentionanycard/pull/11)
at `f979f1af25b5e9e4a3254ec2e195cf681c61e84d`. The image publication is
[run 43898896512](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43898896512).
The controller's `IMAGE` and `SOURCE` pins reference that verified publication.
Its immutable digest is `sha256:972e5e3fbfbc80fde96cbfadfd117e5501fa81d6ce4899295dc8122af632f37d`.
Publication artifact `14241368867` has ZIP SHA-256
`35438684b94adb3bc1a305e4a868da3e072ba0ea80a7ce6a7e8105e7e9e3dcb8`.
The published image passed 156 offline tests, with 25 GPU tests deselected and zero skipped.
The acceptance controller was merged in
[PR #13](https://github.com/jasonjiang8866/flashattentionanycard/pull/13),
at `170aff7513eda7d2a4ed40062e54493b3c816ac6`. Its injected worker and CPU verifier
used that revision; workload code came from the pinned image above.

## Verified result

| Item | Observed evidence |
|---|---|
| Hardware | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb; EUR-IS-2; compute capability 12.0, 46 SMs |
| GPU tests | 25 passed; zero skipped, failed or errored, from `gpu-tests.xml` |
| Search | 18 studies, 36 attempts; three kernels × FP16/BF16 × Grid/Random/TPE; search seed 0 |
| Confirmation | 18 frozen winners × 6 fresh paired workers = 108; every receipt and source-study binding verified |
| Acceptance | `status=accepted`, `archive_verified=true`; retrieved archive independently passed `runpod.verify_paid.verify_archive` |
| Software | PyTorch 2.9.0+cu128, Triton 3.5.0, Optuna 4.9.0 |
| Config SHA-256 | `b56bd7d89f9d95f898df26432d09b1f836e9ee82597b4b44cf7756b8870f0453` |
| Suite active time | 497.504400 seconds, including bundle creation; excludes allocation/image startup and final cleanup |
| Cleanup | Original and independent recovery each recorded three consecutive absence checks; no persistent volumes created |

The provision record identifies Pod `kb8xh7ox1us59a`, owned by
`faac-20261103-usd1-43917307722`; both cleanup records match that owner name.
Recovery's `creation_confirmed=false` means its
fresh runner had no local creation intent; the original record confirms creation.
It is not evidence that the run created no Pod.

### Paired measurements

All 18 rows are preserved byte-for-byte in
[`evidence/m1-20261117-baseline-summary.csv`](evidence/m1-20261117-baseline-summary.csv).
Values below are the median of six within-pair **PyTorch / candidate** latency ratios;
greater than 1 means the candidate was faster. The CSV also retains latencies,
search-best values and descriptive 95% bootstrap intervals.

| Workload | Grid winner | TPE winner | Random winner |
|---|---:|---:|---:|
| GEMM FP16 | 0.959× | 0.956× | 0.938× |
| GEMM BF16 | 1.332× | 1.370× | 1.319× |
| RMSNorm FP16 | 3.132× | 3.285× | 3.368× |
| RMSNorm BF16 | 3.064× | 3.311× | 3.100× |
| MoE grouped GEMM FP16 | 1.745× | 1.743× | 1.777× |
| MoE grouped GEMM BF16 | 1.962× | 2.008× | 1.988× |

GEMM uses `(127,193,257)`, RMSNorm `(127,193)`, and MoE uses N=193, K=257
with expert tokens `[0,1,17,65]`. These are tiny synthetic, hot-buffer measurements
on one device/session with one search seed. They do not rank the search methods,
establish production throughput, or demonstrate cross-GPU transfer. GEMM FP16
regressed against PyTorch; retain that result. MoE's baseline is eager per-expert
`torch.mm`, not an optimized grouped-GEMM library. See [MEASUREMENT.md](MEASUREMENT.md).

### Retrieved artifacts

The original ZIPs were downloaded again on 2026-11-27 SGT. Their SHA-256 values
matched GitHub's artifact digests; the inner archive's receipts, raw pairs and
sample-derived CSV statistics were independently reverified locally without a GPU.

| Artifact | ID | SHA-256 |
|---|---|---|
| Acceptance ZIP | `14254340944` | `940f8e9b9dabaf60f27fefcc9d4ae5f6ad58b8fb53f4a3f7c4ea655e300e4b9b` |
| Independent recovery ZIP | `14252468356` | `613550d55907baf57271f7477c2765dc8541bc62c352ff322f11d837ff8b8734` |
| Retained authorization marker ZIP | `14250806483` | `77a687e5bc8a463eb5c63d9da2fc853e038378d586e2a28f01c602a46a2f3d76` |
| Inner `smoke.tar.gz` | 2,133,466 bytes | `2f62441fbe4016e1246ea24e4550fbc57acb8606c15193e0f908b5eab2d3503e` |

The marker's `run-id.txt` contains exactly `43917307722`. Acceptance/recovery ZIPs
are scheduled to expire on 2026-12-16 UTC; the marker expires on 2027-02-14 UTC.
These links are historical evidence, not permanent artifact hosting. The committed
CSV is a report copy; full receipt/raw-pair verification requires the original archive.

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

## Budget and run reconciliation

The user's project allowance was $10 / 500 minutes. The accepted attempt retained the
stricter existing **$1 cumulative controller cap**, including earlier reserved spend,
temporary disk, retries already reconciled and cleanup. It neither resets the
ledger nor allocates the entire project allowance.

M0 run `43743010737` consumed marker `14173177303`; retain it and reconcile only
that exact pair. Re-downloaded original/recovery ZIP hashes matched the records in
[GPU_SMOKE.md](GPU_SMOKE.md). Both cleanup records confirmed absence three times.
Create intent `1794764824.8761477` to independent absence `1794765143.290164`
is 318.414016 seconds. At the recorded $0.69/hour plus temporary disk rate this
reserves $0.061555832, bringing the earlier $0.16 reserve to $0.221555832.
The cumulative reserve before M1 was rounded up to **$0.23**.

M1 run `43917307722` consumed marker `14250806483`; reconcile only that exact pair.
Creation intent `1794849661.7757976` through independent absence
`1794850584.8404558` spans **923.064658 seconds** (about 15 minutes 23 seconds).
At the recorded GPU rate of $0.69/hour plus $0.005952381/hour for temporary disk,
the incremental conservative estimate is **$0.178446957**. Adding the prior $0.23
reserve gives $0.408446957; round the cumulative reserve up to **$0.41**.
This is an intent-to-independent-absence estimate, not a provider invoice.
The original cleanup's earlier $0.399343762 cumulative estimate is superseded by
this more conservative interval. No marker is deleted; unknown attempts still block.

The single-Pod work window remains at most 1,200 seconds, with 360 seconds reserved
for cleanup and 90 seconds of the work window reserved for result transfer. The
controller shortens the work window when necessary to keep the planned cumulative
cost at or below $0.80, leaving at least $0.20 unallocated under its $1 cap.
With the reconciled $0.41 reserve, a hypothetical $1.50/hour GPU quote leaves a
572-second work window; a $0.69/hour quote leaves 1,200 seconds. These are arithmetic
examples using historical rates, not current quotes or permission to launch.
They are admission estimates, not provider-enforced caps.
Fresh API quotes/capacity, monthly guard and exact image/source equality still apply.
The GPU allowlist, CUDA minimums, $1.50/hour ceiling, 40 GB ephemeral disk,
one-create/no-automatic-retry policy and independent cleanup are unchanged.

## Execution status

The existing cutoff **2026-11-23 00:00 UTC / 08:00 SGT has expired**. This update
reconciles completed work; it does not extend the cutoff, reset the ledger, publish
an image or start a Pod. The $10 / 500-minute project allowance does not silently
replace the controller's $1 cap or authorize a different experiment.

Before another paid experiment, agree its scope/window and budget accounting,
prepare its bounded controller and verifier, then use fresh image/source, quote,
capacity and cleanup evidence. The existing `runpod-gpu-smoke.yml` is fixed to
the accepted 18-study smoke; it is not a paid pilot launcher.

## Next development slice

| Order | Deliverable | Acceptance gate |
|---|---|---|
| 1 | Close M1: bilingual progress, evidence, exact marker and $0.41 reserve | Reverify the downloaded archive; exact-marker, budget and expired-window regression tests pass |
| 2 | Prepare a bounded single-GPU pilot controller and plan-aware archive verifier | Support the existing pilot plan with 12 studies, 216 attempts and 72 paired workers; reject incomplete/mismatched plans and keep cleanup on every exit |
| 3 | Add phase/progress reporting and budget breakdown | Distinguish allocation/image startup, GPU tests, search, confirmation, download and cleanup; account for startup and 72 additional workers before admission |
| 4 | Execute a newly scoped pilot, then decide on cross-GPU transfer | First validate larger shapes on one approved GPU; only then prepare the matched two-GPU cold/warm comparison with source costs included |

Reuse [PILOT.md](PILOT.md)'s FP16 GEMM 2048³ and RMSNorm 4096×4096,
search seeds `[0,1]`, equal method budgets and six independent confirmation pairs
per study. Two seeds remain exploratory. RTX 4090 is the operational first choice
when authorized, available and within a fresh quote; a 3090/5090 comparison has a
different architecture objective and must identify its actual source/target devices.
No large-shape or transfer performance result is claimed yet.

CPU-only planning is available now:

```bash
python -m runpod.launch pilot --confirm-repetitions 6 --dry-run
python -m runpod.launch pilot --confirm-repetitions 6 --dry-run \
  --source-run /tmp/imported/3090-pilot --source-seed 0 --top-k 3 \
  --output /tmp/5090-pilot-transfer
```

Expected cold plan: 12 studies / 216 attempts / 72 paired workers.
Expected target cold+warm plan: 16 / 288 / 96. Worker-timeout sums are not expected
runtime or billing caps. Read-only planning neither inspects CUDA nor creates compute.
