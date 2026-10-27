# bo-kernel-tuner

基于 **Optuna multivariate TPE + Triton** 的硬件感知 GPU 算子调优与跨架构迁移实验框架。
本项目位于 `flashattentionanycard` 仓库；当前实现 GEMM、Fused RMSNorm 和 MoE grouped GEMM。
目标平台为 Linux + NVIDIA CUDA、compute capability 8.0 及以上，包括 Ampere 与 Blackwell。

**验证状态：CPU 回归与离线代码生成可自动验证；真实 GPU 精度、吞吐和迁移收益必须在目标 GPU 上测量。仓库不包含虚构的 GPU benchmark 结果。**

## RunPod 一键实验

已提供 [RunPod 脚本、固定依赖 Dockerfile 和完整操作说明](runpod/README.md)。
在 RunPod GPU Pod 中检出本分支后执行：

```bash
bash runpod/bootstrap.sh smoke --output /workspace/bo-kernel-tuner/results/smoke-01
```

默认验证全部 GPU 测试，并完成三个算子的 FP16/BF16 小规模对照、出图和结果打包。
`performance`、跨 GPU warm-start 及经过产物校验的 study 级恢复见上述文档。

## 安装

使用独立 Python 3.10+ 环境。安装与驱动兼容、包含目标 GPU 架构的 PyTorch CUDA wheel，再安装本项目。
建议两台机器使用相同 Python、PyTorch、Triton、Optuna 版本，并保存完整环境。

```bash
git clone https://github.com/jasonjiang8866/flashattentionanycard.git
cd flashattentionanycard
# PR 尚未合并时使用实现分支：
git switch feat/runpod-experiments
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip freeze > environment-lock.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

开发依赖范围见 `pyproject.toml`。初始离线代码生成测试固定 Triton 3.5.0；CPU CI 覆盖 Optuna 4.5.0/4.9.0。
这些是验证基线，不代表任意驱动/软件组合已通过 GPU 验证。

## 目录

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

`worker.py` 隔离 CUDA context。单个候选的编译错误、OOM、精度失败、进程崩溃或超时不会作为成功测量写入数据库，也不会污染后续 trial。

## 快速验证

```bash
# 不加载 CUDA，不生成性能数据：
python -m bo_kernel_tuner.run_benchmark --dry-run --kernel gemm

# CPU 回归；不安装 GPU 依赖也可单独执行：
python -m pytest -q -m 'not gpu and not compile'

# Triton 离线编译到 sm80/sm86/sm120；不执行 kernel：
python -m pytest -q bo_kernel_tuner/tests/test_compile.py

# 在 GPU 机器上执行真实数值/计时检查：
python -m pytest -q -m gpu

# 小型端到端冒烟：三个搜索器、同一空间、同一 attempt 预算。
python -m bo_kernel_tuner.run_benchmark \
  --kernel gemm --device 0 --profile smoke --mode compare \
  --m 127 --n 193 --k 257 --seeds 0 --trials 4 --startup-trials 2 \
  --output results/smoke-gemm
```

GPU correctness 测试覆盖 FP16/BF16、非 tile 整数倍、单元素、空 expert、不均匀 expert 和多种 launch 配置。
每个正式 trial 也必须先通过默认两组输入 seed 的 FP32 reference 校验，才允许进入计时。

## 同一 GPU 上比较搜索器

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

`--mode tune --method grid|random|tpe` 运行单个搜索器。
`--time-budget 600` 增加每个 study 的端到端秒数预算；`--trial-timeout 180` 限制每个子进程。
数据库提交与终止清理可能使实际总耗时略超预算，实际耗时会完整记录。

每个 study 使用单个 GPU、`n_jobs=1`。每个 seed 内搜索器顺序随机化，以减轻固定执行顺序引起的温度漂移；
同时运行其他 GPU 工作仍会干扰测量，应在独占、稳定的环境下实验。

### 搜索空间和公平性

| 算子 | 可调参数 | full 空间 |
|---|---|---:|
| GEMM | BLOCK_M/N/K、GROUP_M、num_warps、num_stages | 432 |
| RMSNorm | ROWS_PER_CTA、num_warps | 12 |
| MoE grouped GEMM | BLOCK_M/N/K、CTA_MULTIPLIER、num_warps、num_stages | 432 |

三种方法使用**完全相同的 categorical domain**。GridSampler 以 seed 随机排列笛卡尔积，避免词典序偏差；
随机搜索有放回采样，TPE 也允许重复，重复配置会重新测量并标记。
有效 attempt 预算统一为 `min(--trials, grid_size)`，避免 Grid 已穷尽后让其他方法继续使用更多预算。
失败和静态剪枝同样消耗 attempt。小于完整空间的 Grid 实验属于预算受限的网格前缀；不能称为穷尽基准。

TPE 显式启用 `multivariate=True`。`--startup-trials` 是开始拟合前的随机探索数量；
小空间使用较小 startup 数。RMSNorm 的 12 点空间是有意保留的负对照：网格搜索可能更合适，不能预设 BO 胜出。

硬件感知来自设备属性读取、合法性检查、资源估算与以 SM 数量决定 MoE CTA 数量。
它不是以架构特征为输入的跨设备 surrogate model。TPE 仍为每个 GPU 独立建立模型。

### 资源约束

- 自动读取 GPU 名称、UUID、compute capability、SM 数、显存、每 block shared memory 和线程限制。
- 默认只剪掉明确非法线程配置和 RMSNorm 实现上限。
- shared-memory 公式是 pipeline 的**近似值**，不会默认作为硬约束误删合法配置。
  `--heuristic-smem` 显式开启启发式剪枝，需独立做消融实验。
- 真正的资源可行性由 Triton 编译器和 CUDA launch 判断；成功配置记录编译后的 shared memory、registers 和 spills。
- 没有可靠静态公式能完全预测寄存器占用、编译器布局与 occupancy，代码不会伪造这些上界。

## 3090 → 5090 的 warm-start

两台设备无需在同一主机。复制源端 `result.json` 即可，目标端必须使用相同代码、搜索空间和 workload（含 dtype、shape、expert 分布及 data seed）。
GPU 编号是当前进程中、经过 `CUDA_VISIBLE_DEVICES` 映射后的逻辑编号。

```bash
# 在 3090 上运行前面的 bf16 GEMM 比较，得到 ampere-gemm/tpe-seed-0/result.json。
# 在 5090 上，使用同一代码版本并复制该文件：
python -m bo_kernel_tuner.run_benchmark \
  --kernel gemm --dtype bf16 --device 0 --mode compare \
  --m 2048 --n 2048 --k 2048 --trials 100 --seeds 0,1,2,3,4 \
  --warm-start results/ampere-gemm/tpe-seed-0/result.json --top-k 5 \
  --output results/blackwell-gemm-transfer
```

目标端会比较 Grid、Random、冷启动 TPE、warm-start TPE。
迁移模块对源 GPU 完成且数值合法的 trial 按 latency 排序，去重并检查目标 GPU 的静态约束，再 `enqueue_trial`。
**不通过 `add_trial` 导入源 GPU 的目标值。** 排队参数在目标 GPU 上重新编译、校验和计时，并计入目标预算。
源文件 SHA-256、源/目标硬件和候选数量会写入 provenance。

这是参数排名先验/初始化迁移，并不是把源端 TPE posterior 原样搬到目标 GPU。
初始 warm trial 也参与目标端 TPE 的后续训练。若 warm 数少于 startup 数，会继续随机探索直到达到 startup 阈值。

为避免只挑一个幸运源 seed，正式实验应预先固定源 study，或对源 seed 分别重复整组迁移。
warm-start 曲线默认只统计目标端开销；源 GPU 搜索成本需要单列，计算 amortized savings 时也要计入。
反向 5090 → 3090 迁移应独立执行，以检验迁移的不对称性。

## 结果、图表和统计

```bash
python -m bo_kernel_tuner.analysis.plot_metrics \
  results/ampere-gemm results/blackwell-gemm-transfer \
  --output results/figures --target-factor 1.05
```

每个 study 输出：

| 文件 | 内容 |
|---|---|
| `study.sqlite3` | Optuna trial 状态、参数、measurement 和实验 metadata |
| `result.json` | 原子替换的实验快照、最佳有效配置和全部 attempted trials |
| `trials.csv` | 便于 pandas/R 读取的逐 trial 表 |
| `logs/trial-N.log` | 隔离 worker 的编译、CUDA 或精度失败日志 |

结果目录拒绝覆盖。每个 trial 后保存快照；被中断的实验标记 `interrupted`，当前版本不提供自动 resume。
全部候选失败时没有 best，CLI 返回非零状态，不会生成“成功”的调优结果。

分析模块按硬件、workload、代码、依赖版本、测量协议、预算和剪枝选项划分 cohort。
不同 GPU 的收敛曲线不会混合聚合。相同 cohort 内要求各方法 seed 集一致且不重复。

- `convergence.pdf/png`：按 attempt 和端到端时间绘制 best-so-far，中位数及跨 seed IQR。
  使用共同观测区间；在所有 seed 都有有效 incumbent 前不画聚合值，避免只保留成功 seed 的偏差。
- `tuning_time.pdf/png`：总调优时间与每个 seed 的观测点。
- `summary.csv`：成功/失败/剪枝/唯一配置计数、最佳耗时、到达目标的 attempt 与秒数。
- `paired_comparisons.csv`：配对 seed 的 baseline/candidate 比值，大于 1 表示 candidate 更快；
  至少 3 对时输出固定随机种子的 5,000 次 bootstrap percentile 95% CI。
  任一配对未达到目标则不计算该指标的比值，保留 hit rate 与缺失值，避免删掉失败样本后夸大加速。
- `parameter_shift.pdf/png`：不同硬件/seed 的最佳参数热力图，颜色表示候选序号归一化值，数字表示真实参数。
  它是参数偏移的描述图，不把离散候选间距离解释成物理距离。

目标阈值默认是 cohort 中**已观察到的最优 latency × 1.05**，不是已知全局最优。
未到达目标的 run 作为右删失样本保留。以上 CI 与阈值比较用于探索性分析；论文中的确认性结论应使用预先定义的目标与独立重复实验。

### 测量协议

1. 每个候选在独立进程与空 Triton/CUDA 磁盘缓存中编译，重复候选也遵守相同规则。
2. FP32 reference 禁用 TF32。FP16 使用 `rtol=0.01, atol=0.02`；BF16 使用 `rtol=0.02, atol=0.15`。
   两组随机输入、NaN 填充输出可检测漏写和非有限值，日志记录最大绝对误差。容差需结合真实 workload 再评估。
3. 预热后捕获默认 32 次 launch 的 CUDA Graph；30 次 replay 分别用 CUDA events 计时并除以 batch。
   objective 为中位数，保留全部 sample、MAD、最小/最大值。
4. 计时排除 Python dispatch、输入生成、reference 和编译；端到端 search wall time 则包括这些成本、进程启动、Optuna 和检查点开销。
5. 同一输入/输出 buffer 重用，属于 hot-buffer/cache 协议。小 RMSNorm 可能主要受 cache/launch 限制，不能把结果直接称为 DRAM 带宽。

最佳配置的统计量存在 winner's curse。可将该 study 作为 `--warm-start`，以 `--top-k 1 --trials 1 --mode tune --method tpe`
在新的输出目录重新测量；论文应进一步进行独立重复、不同输入及真实 shape 验证。

## 建议实验矩阵

| 维度 | 初始设置 |
|---|---|
| 架构 | RTX 3090 / Ampere，RTX 5090 / Blackwell，记录真实硬件 metadata |
| dtype | FP16、BF16 分开分析 |
| GEMM | `(1024,1024,1024)`、`(2048,2048,2048)`、`(127,193,257)`、瘦长矩阵 |
| RMSNorm | rows 4096，hidden 1024/4096/8192，另测非 2 次幂 hidden |
| MoE | balanced、skewed、包含空 expert 的相同总 token 数 workload |
| search seed | 至少 5 个，正式结论建议增加独立重复 |
| 预算 | GEMM/MoE 25/50/100；RMSNorm 最多 12；另做同 wall-time 预算实验 |
| transfer | cold、top-1/5/10 warm，两个方向分别测试 |
| 消融 | 默认硬约束 vs 启发式 shared-memory 剪枝 |

不应预设 TPE 更快。报告失败率、负迁移和无显著收益的 workload；只在完整网格全部测量后，才能讨论该离散空间内的经验最优值。

当前内核未覆盖 FlashAttention、backward、FP8、AMD/Intel、非 contiguous layout、MoE routing/scatter 和 Blackwell 专用 TMA/warp-specialization。
这里只比较搜索方法找到此实现最佳配置的效率，不宣称超过 cuBLAS 或厂商最优算子。

## 参考

- [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Optuna GridSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html)
- [Optuna enqueue_trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.enqueue_trial)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
