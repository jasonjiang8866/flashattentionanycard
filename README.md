# bo-kernel-tuner

**English** | [Simplified Chinese](README.zh-CN.md)

A hardware-aware GPU kernel tuning and cross-architecture transfer experiment framework built on **Optuna multivariate TPE + Triton**.
Hosted in the `flashattentionanycard` repository, the project currently implements GEMM, Fused RMSNorm, and MoE grouped GEMM.
It targets Linux + NVIDIA CUDA with compute capability 8.0 or newer, including Ampere and Blackwell.

## Project progress

Updated **2026-11-27 (SGT)**. **M1 passed real-GPU smoke acceptance on 2026-11-17 (SGT)**:
25 GPU tests passed with zero skips, 18 search studies completed, and all 18 frozen winners
completed six independent candidate/PyTorch pairs each (**108 paired workers**).

| Milestone | Verified status | Evidence / next deliverable |
|---|---|---|
| M0: execution pipeline | Accepted on PRO 6000 Blackwell MIG 24GB; 19 GPU tests, 18 studies, 36 attempts | [M0 run](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43743010737) |
| M1: PyTorch baselines and independent confirmation | Accepted on the same GPU model; downloaded archive and sample-derived summaries independently verified | [M1 run](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43917307722), [independent cleanup](https://github.com/jasonjiang8866/flashattentionanycard/actions/runs/43920030853) |
| Next: larger-shape and cross-GPU pilot | Planned; no pilot performance or transfer result yet | [Pilot protocol](runpod/PILOT.md), [development sequence and gates](runpod/M1_ACCEPTANCE.md#next-development-slice) |

The M1 temporary Pod was deleted and independently confirmed absent. Its conservative
incremental GPU + temporary-disk estimate is **$0.18**, bringing the cumulative reserved
estimate to **$0.41**; these are estimates, not provider invoices. The controller's existing
execution window expired on **2026-11-23 08:00 SGT**; ledger reconciliation does not renew it.

This acceptance covers small synthetic workloads on one device/session. It does not establish
cross-GPU portability, TPE superiority, transfer benefits, or production-scale speedups.
GEMM FP16 was slower than its PyTorch baseline in this smoke; all cases, including regressions,
remain in the [acceptance record](runpod/M1_ACCEPTANCE.md).
See the [M1 measurement protocol](runpod/MEASUREMENT.md) for frozen winners, held-out input seeds,
alternating execution order, raw samples and statistical limitations.

## One-command RunPod experiments

[RunPod scripts, a Dockerfile with pinned dependencies, and full operating instructions](runpod/README.md) are available.
After checking out `main` in a RunPod GPU Pod, run:

```bash
bash runpod/bootstrap.sh smoke --output /workspace/bo-kernel-tuner/results/smoke-01
```

By default, this runs all GPU tests, small FP16/BF16 comparisons for all three kernels, plot generation, and result packaging.
See the linked documentation for `performance`, cross-GPU warm-start, and study-level recovery with artifact validation.

## Installation

Use a dedicated Python 3.10+ environment. Install a PyTorch CUDA wheel that supports both your driver and target GPU architecture, then install this project.
Use the same Python, PyTorch, Triton, and Optuna versions on both machines, and record the complete environment.

```bash
git clone https://github.com/jasonjiang8866/flashattentionanycard.git
cd flashattentionanycard
git switch main
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip freeze > environment-lock.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

See `pyproject.toml` for development dependency ranges. Initial offline code-generation tests pin Triton 3.5.0; CPU CI covers Optuna 4.5.0/4.9.0.
These are validation baselines, not evidence that every driver/software combination has passed GPU validation.

## Repository layout

```text
bo_kernel_tuner/
├── configs/search_spaces.py
├── kernels/
│   ├── __init__.py
│   ├── gemm.py
│   ├── rmsnorm.py
│   └── moe_grouped_gemm.py
├── tuner/
│   ├── __init__.py
│   ├── hardware.py
│   ├── engine.py
│   ├── transfer.py
│   └── worker.py
├── analysis/plot_metrics.py
├── tests/
│   ├── test_correctness.py
│   ├── test_compile.py
│   ├── test_engine.py
│   └── test_analysis.py
└── run_benchmark.py
```

`worker.py` isolates the CUDA context. A candidate's compilation error, OOM, correctness failure, process crash, or timeout is never recorded as a successful measurement and does not contaminate later trials.

## Quick validation

```bash
# No CUDA loading and no performance data:
python -m bo_kernel_tuner.run_benchmark --dry-run --kernel gemm

# CPU regression tests; these can also run without installing GPU dependencies:
python -m pytest -q -m 'not gpu and not compile'

# Compile Triton offline for sm80/sm86/sm120 without executing kernels:
python -m pytest -q bo_kernel_tuner/tests/test_compile.py

# Run real numerical and timing checks on a GPU machine:
python -m pytest -q -m gpu

# Small end-to-end smoke test: three search methods, one space, one attempt budget.
python -m bo_kernel_tuner.run_benchmark \
  --kernel gemm --device 0 --profile smoke --mode compare \
  --m 127 --n 193 --k 257 --seeds 0 --trials 4 --startup-trials 2 \
  --output results/smoke-gemm
```

GPU correctness tests cover FP16/BF16, dimensions that are not multiples of tile sizes, single elements, empty experts, uneven expert workloads, and multiple launch configurations.
Every benchmark trial must also pass FP32 reference checks with two input seeds by default before timing is allowed.

## Compare search methods on one GPU

```bash
python -m bo_kernel_tuner.run_benchmark \
  --kernel gemm --mode compare --dtype bf16 --device 0 \
  --m 2048 --n 2048 --k 2048 --trials 100 \
  --seeds 0,1,2,3,4 --output results/ampere-gemm

python -m bo_kernel_tuner.run_benchmark \
  --kernel rmsnorm --mode compare --m 4096 --n 4096 \
  --trials 12 --startup-trials 3 --seeds 0,1,2,3,4 \
  --output results/ampere-rmsnorm

python -m bo_kernel_tuner.run_benchmark \
  --kernel moe_grouped_gemm --mode compare --n 2048 --k 1024 \
  --expert-tokens 0,16,64,256,1024,32,8,128 --trials 100 \
  --seeds 0,1,2,3,4 --output results/ampere-moe
```

`--mode tune --method grid|random|tpe` runs a single search method.
`--time-budget 600` adds an end-to-end budget in seconds for each study; `--trial-timeout 180` limits each subprocess.
Database commits and termination cleanup may slightly exceed the budget; actual elapsed time is recorded in full.

Each study uses one GPU with `n_jobs=1`. Search-method order is randomized within each seed to reduce thermal drift caused by a fixed execution order.
Other concurrent GPU work still interferes with measurements, so use an exclusive, stable environment.

### Search spaces and fairness

| Kernel | Tunable parameters | Full space |
|---|---|---:|
| GEMM | BLOCK_M/N/K, GROUP_M, num_warps, num_stages | 432 |
| RMSNorm | ROWS_PER_CTA, num_warps | 12 |
| MoE grouped GEMM | BLOCK_M/N/K, CTA_MULTIPLIER, num_warps, num_stages | 432 |

All three methods use **exactly the same categorical domains**. GridSampler shuffles the Cartesian product using the seed to avoid lexicographic bias.
Random search samples with replacement, and TPE also permits duplicates; repeated configurations are remeasured and flagged.
The effective attempt budget is uniformly `min(--trials, grid_size)`, preventing other methods from spending more attempts after Grid exhausts the space.
Failures and static pruning also consume attempts. Grid experiments below the full-space budget are budget-limited grid prefixes, not exhaustive baselines.

TPE explicitly enables `multivariate=True`. `--startup-trials` sets the number of random exploration trials before model fitting begins;
use a smaller startup count for small spaces. RMSNorm's 12-point space is an intentional negative control: grid search may be more suitable, and BO must not be assumed to win.

Hardware awareness comes from reading device properties, checking legality, estimating resources, and using the SM count to determine the number of MoE CTAs.
This is not a cross-device surrogate model that takes architecture features as inputs. TPE still builds a separate model for each GPU.

### Resource constraints

- Automatically reads the GPU name, UUID, compute capability, SM count, VRAM, per-block shared memory, and thread limits.
- By default, prunes only clearly illegal thread configurations and configurations exceeding the RMSNorm implementation's limits.
- The shared-memory formula is a **pipeline approximation** and is not a hard constraint by default, avoiding accidental removal of legal configurations.
  `--heuristic-smem` explicitly enables heuristic pruning, which requires a separate ablation study.
- Actual resource feasibility is determined by the Triton compiler and CUDA launch; successful configurations record compiled shared memory, registers, and spills.
- No reliable static formula fully predicts register usage, compiler layouts, and occupancy; the code does not fabricate such bounds.

## Warm-start from a 3090 to a 5090

The two devices do not need to be in the same host. Copy the source `result.json`; the target must use the same code, search space, and workload, including dtype, shape, expert distribution, and data seed.
GPU indices are logical indices within the current process, after mapping through `CUDA_VISIBLE_DEVICES`.

```bash
# Run the bf16 GEMM comparison above on the 3090 to produce ampere-gemm/tpe-seed-0/result.json.
# On the 5090, use the same code revision and copy that file:
python -m bo_kernel_tuner.run_benchmark \
  --kernel gemm --dtype bf16 --device 0 --mode compare \
  --m 2048 --n 2048 --k 2048 --trials 100 --seeds 0,1,2,3,4 \
  --warm-start results/ampere-gemm/tpe-seed-0/result.json --top-k 5 \
  --output results/blackwell-gemm-transfer
```

The target compares Grid, Random, cold-start TPE, and warm-start TPE.
The transfer module ranks completed, numerically valid source-GPU trials by latency, deduplicates them, checks the target GPU's static constraints, and then calls `enqueue_trial`.
**Source-GPU objective values are not imported through `add_trial`.** Queued parameters are recompiled, validated, and timed on the target GPU, counting toward its budget.
The source file's SHA-256, source/target hardware, and candidate counts are recorded in provenance.

This transfers a parameter-ranking prior/initialization, not the source TPE posterior unchanged to the target GPU.
Initial warm trials also train the target TPE model. If there are fewer warm trials than startup trials, random exploration continues until the startup threshold is reached.

To avoid selecting a single lucky source seed, formal experiments should fix the source study in advance or repeat the full transfer experiment for each source seed.
Warm-start curves count only target-side costs by default; source-GPU search cost must be reported separately and included when computing amortized savings.
Run the reverse 5090 → 3090 transfer independently to test transfer asymmetry.

## Results, plots, and statistics

```bash
python -m bo_kernel_tuner.analysis.plot_metrics \
  results/ampere-gemm results/blackwell-gemm-transfer \
  --output results/figures --target-factor 1.05
```

Each study writes:

| File | Contents |
|---|---|
| `study.sqlite3` | Optuna trial states, parameters, measurements, and experiment metadata |
| `result.json` | Atomically replaced experiment snapshot, best valid configuration, and all attempted trials |
| `trials.csv` | Per-trial table for pandas/R |
| `logs/trial-N.log` | Compilation, CUDA, or correctness failure logs from the isolated worker |

Existing result directories cannot be overwritten. A snapshot is saved after every trial; interrupted experiments are marked `interrupted`, and the current version does not automatically resume them.
If every candidate fails, there is no best configuration and the CLI returns a nonzero status instead of producing a "successful" tuning result.

The analysis module groups results into cohorts by hardware, workload, code, dependency versions, measurement protocol, budget, and pruning options.
Convergence curves from different GPUs are never aggregated together. Within a cohort, all methods must have matching, nonduplicate seed sets.

- `convergence.pdf/png`: best-so-far curves by attempt and end-to-end time, with medians and cross-seed IQR.
  Uses a common observation interval; aggregates are withheld until every seed has a valid incumbent, avoiding bias from retaining only successful seeds.
- `tuning_time.pdf/png`: total tuning time and observations for each seed.
- `summary.csv`: counts of successful/failed/pruned/unique configurations, best latency, and attempts and seconds to reach the target.
- `paired_comparisons.csv`: baseline/candidate ratios for paired seeds; values above 1 mean the candidate is faster.
  With at least 3 pairs, reports a 95% bootstrap percentile CI from 5,000 resamples using a fixed random seed.
  If either run in a pair misses the target, that metric's ratio is not computed; hit rates and missing values are retained to avoid overstating speedups by dropping failures.
- `parameter_shift.pdf/png`: heatmap of best parameters across hardware/seeds; colors encode normalized candidate indices, while numbers show actual parameter values.
  This describes parameter shifts without interpreting distances between discrete candidates as physical distances.

The default target threshold is the **best observed latency in the cohort × 1.05**, not a known global optimum.
Runs that miss the target are retained as right-censored observations. These CIs and threshold comparisons support exploratory analysis; confirmatory claims in a paper should use predefined targets and independent repeated experiments.

### Measurement protocol

1. Every candidate is compiled in a separate process with empty Triton/CUDA disk caches; repeated candidates follow the same rules.
2. The FP32 reference disables TF32. FP16 uses `rtol=0.01, atol=0.02`; BF16 uses `rtol=0.02, atol=0.15`.
   Two random inputs and NaN-filled outputs help detect unwritten elements and nonfinite values; logs record the maximum absolute error. Tolerances should be reassessed for real workloads.
3. After warmup on a side stream, a CUDA Graph captures 32 launches by default; 30 replays are individually timed with CUDA events and divided by the batch size.
   The objective is the median; all samples, MAD, minimum, and maximum are retained.
4. Timing excludes Python dispatch, input generation, reference computation, and compilation. End-to-end search wall time includes these costs, process startup, Optuna, and checkpoint overhead.
5. Input/output buffers are reused, making this a hot-buffer/cache protocol. Small RMSNorm workloads may be cache/launch limited, so their results cannot be directly described as DRAM bandwidth.

The search statistic for the best configuration is subject to the winner's curse. Use [M1 independent confirmation](runpod/MEASUREMENT.md) to freeze that configuration and measure it against the PyTorch baseline in paired runs,
reporting the search best and confirmation values on new inputs separately. Replay samples are not independent experimental repeats; papers should also validate with independent sessions and real workload shapes.

## Suggested experiment matrix

| Dimension | Initial settings |
|---|---|
| Architecture | RTX 3090 / Ampere, RTX 5090 / Blackwell; record actual hardware metadata |
| dtype | Analyze FP16 and BF16 separately |
| GEMM | `(1024,1024,1024)`, `(2048,2048,2048)`, `(127,193,257)`, and tall/skinny matrices |
| RMSNorm | 4096 rows, hidden sizes 1024/4096/8192, plus non-power-of-two hidden sizes |
| MoE | Balanced, skewed, and empty-expert workloads with the same total token count |
| Search seeds | At least 5; add independent repeats for formal conclusions |
| Budget | GEMM/MoE: 25/50/100; RMSNorm: at most 12; also run equal-wall-time-budget experiments |
| Transfer | Cold, top-1/5/10 warm; test both directions separately |
| Ablation | Default hard constraints vs. heuristic shared-memory pruning |

Do not assume TPE is faster. Report failure rates, negative transfer, and workloads with no significant benefit; only after measuring the complete grid can you discuss the empirical optimum within that discrete space.

Current kernels do not cover FlashAttention, backward, FP8, AMD/Intel, noncontiguous layouts, MoE routing/scatter, or Blackwell-specific TMA/warp-specialization.
The project compares how efficiently search methods find the best configuration of these implementations; it does not claim to outperform cuBLAS or vendor-optimized kernels.

## References

- [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Optuna GridSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html)
- [Optuna enqueue_trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.enqueue_trial)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
