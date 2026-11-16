# M1: PyTorch baselines and independent winner confirmation

M0 passed on 2026-11-16 SGT: [GPU run 43743010737](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43743010737),
19 GPU tests, 18 studies and 36 attempts on PRO 6000 Blackwell MIG 24GB.
The [independent cleanup run](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43743491986)
confirmed the temporary Pod absent. This establishes the old smoke execution path;
it is not a performance comparison or real-GPU validation of the new M1 code.

## What is compared

| Workload | Baseline | Scope |
| --- | --- | --- |
| GEMM | `torch.mm(a, b, out=out)` | Same input dtype, shapes and preallocated output |
| RMSNorm | `torch.nn.functional.rms_norm` | Same input dtype, weight and explicit epsilon; native output allocation |
| MoE grouped GEMM | One `torch.mm(..., out=...)` per nonempty expert | Prebuilt views; same ragged tokens, no routing/scatter or padded work |

The MoE baseline is an eager per-expert implementation, **not an optimized grouped-GEMM
library**. RMSNorm includes the native operation's GPU intermediates. CUDA Graph capture
manages its allocations; Python dispatch/allocation overhead is outside event timing.
These comparisons do not establish superiority over every vendor implementation, an
end-to-end model speedup, or DRAM bandwidth. Reused buffers may fit in cache.

The benchmark generates equal input values from the same device RNG seed for both
implementations, using separate buffers. Both must pass the existing FP32-reference
checks on every validation seed. TF32 and FP16/BF16 reduced-precision matmul reductions
are explicitly disabled. GEMM/MoE output remains FP16/BF16; reference output is FP32.
Tolerances remain FP16 `rtol=0.01, atol=0.02`, BF16 `rtol=0.02, atol=0.15`.

## Measurement and selection protocol

1. Search still compares Grid/Random/TPE with matched attempt budgets. Failed and pruned
   attempts consume budget. The search winner is frozen before confirmation starts.
2. `cuda_graph_v2` warms up on a side stream, captures `batch` calls, then measures each
   replay with CUDA events and divides by `batch`. Preserve every sample, median, MAD,
   minimum and maximum. Reject nonpositive, NaN or infinite samples.
3. Confirmation uses the source study's warmup/repeats/batch/validation counts, on the
   same GPU UUID and software versions. Each pair runs in a fresh subprocess with fresh
   Triton/CUDA disk-cache directories. No parameter search occurs during confirmation.
4. Six paired repetitions are the default. Candidate/PyTorch ordering alternates AB/BA.
   With an odd repetition count, order counts differ by one. At least three pairs are required.
5. Search input seeds are `data_seed .. data_seed + validation_seeds - 1`. Pair `i` starts
   at `data_seed + (i + 1) * validation_seeds`; pair seed ranges never overlap search or
   each other. The first seed in each pair is timed. Shapes and the input distribution stay fixed.
6. Speedup is PyTorch median / candidate median **within each pair**. Report the median
   paired ratio and a seeded 5,000-resample percentile interval over independent worker
   pairs. Inner CUDA replay samples are not extra independent observations.

This interval describes a frozen configuration on one device/session across the declared
inputs and process repetitions. It is exploratory, especially with only 3–6 pairs. It
does not measure search-seed variability, cross-session variation or fleet-wide performance.
Do not select the best confirmation repetition or retry failures until they disappear.
Search-best latency remains a separate field; confirmation does not reselect a winner.

## Run in an already prepared GPU environment

The [M1 image publication](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43898896512)
contains this protocol at `f979f1af25b5e9e4a3254ec2e195cf681c61e84d`.
See [M1_ACCEPTANCE.md](M1_ACCEPTANCE.md) for the controller/image handoff, fixed
six-pair acceptance and cost ledger. Publication is offline validation, not real GPU acceptance.

```bash
# CPU-only plan: preserves the 18-study / 36-attempt smoke design and adds 108 pair workers.
python -m runpod.launch smoke --confirm-repetitions 6 --dry-run

# Actual GPU execution; requires separately prepared compute and an external time/cost cap.
python -m runpod.launch smoke --confirm-repetitions 6 \
  --output /workspace/results/m1-smoke

# Larger shapes; inspect the extra confirmation work before choosing a budget.
python -m runpod.launch pilot --confirm-repetitions 6 --dry-run
```

The default `--confirm-repetitions 0` preserves search-only cost. Enabling confirmation
adds one paired worker per repetition **per study**, after all search studies finish.
The plan reports separate search/confirmation worker-timeout sums. These sums are neither
expected durations nor billing caps; GPU tests, orchestration, reports and cleanup add time.
Every enabled confirmation must pass before the suite is marked complete.

A standalone winner can also be confirmed immediately after a current-code search:

```bash
python -m bo_kernel_tuner.tuner.confirmation \
  /workspace/results/search/tpe-seed-0/result.json \
  --output /workspace/results/confirm-tpe-0 --repetitions 6 --dry-run

# Remove --dry-run to execute on the source GPU. --timeout limits each paired worker.
# CPU-only validation after download:
python -m bo_kernel_tuner.tuner.confirmation --verify /workspace/results/confirm-tpe-0
```

Old-code search results are rejected for new confirmation; frozen historical evidence can
be verified without requiring the current code fingerprint. A different GPU requires a
new target search/measurement, not silently treating cross-device timing as confirmation.

## Evidence and failure handling

| File | Evidence |
| --- | --- |
| `source-result.json` | Exact frozen source bytes, including selected trial and search samples |
| `confirmation.json` | Source SHA-256, fixed parameters, device/software/protocol, planned seeds/order, raw pairs and summary |
| `pairs.csv` | One row per completed independent pair, not per CUDA replay |
| `receipt.json` | SHA-256 of the three files above |
| `logs/trial-N.log` | Isolated paired worker diagnostics |
| Suite `baseline-summary.csv` | Case/method/search-seed mapping, search best and confirmed baseline/candidate metrics |

Suite confirmation directories live under `confirmations/cases/<case>/<method>-seed-N/`.
They are included in the existing checksummed archive. The CPU verifier rechecks the frozen
selection, exact plan, sample-derived statistics, numerical-validation records, pair count,
hardware/software identity, CSV, bootstrap summary and receipt.

Failure or interruption retains partial pairs with `summary=null`; it cannot produce a
complete speedup report. The suite also fails if any confirmation fails. Explicit suite
`--resume` revalidates completed confirmations; partial directories are moved to `incomplete/`
before restarting that confirmation. It never appends fresh repetitions to a truncated
series or silently retries within a run. Resume requires unchanged confirmation settings.

The GPU test gate now requires at least **25 tests**: the original 19 plus six paired
baseline/graph tests (three operators × FP16/BF16). CPU numerical tests exercise all
baselines with real PyTorch, including empty experts and ragged/single-element shapes;
they do not claim CUDA graph execution.

M1 GPU acceptance still requires a new-image run with 25 executed GPU tests, zero skips,
all declared studies/confirmations verified, and independently verified Pod cleanup.

## API references (pinned runtime)

- [PyTorch 2.9 CUDA Graph constraints and side-stream warmup](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)
- [PyTorch 2.9 torch.mm](https://docs.pytorch.org/docs/2.9/generated/torch.mm.html)
- [PyTorch 2.9 functional RMSNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.rms_norm.html)
