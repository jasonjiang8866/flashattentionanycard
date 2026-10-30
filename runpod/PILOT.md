# RTX 3090 / RTX 5090 探索性 pilot

`configs/pilot.json` 填补 smoke 与 performance 之间的预算空档。它复用同一 engine、
测量协议、三种 sampler、结果验证和 study 级恢复；默认入口仍为 smoke，performance 配置不变。
本文件是实验方案，**没有真实 CUDA 性能结果**。CPU 测试和 dry-run 不能证明目标硬件可用或更快。

## 固定协议与候选预算

| Workload | Shape / dtype | 搜索空间 | 每方法、每 seed 的 attempts | TPE startup |
|---|---|---:|---:|---:|
| GEMM | M=N=K=2048 / FP16 | full，432 组 | 24 | 6 |
| RMSNorm | M=4096, N=4096 / FP16 | full，12 组 | 12 | 3 |

GEMM 用于观察矩阵乘法 tile、warp、pipeline 参数；RMSNorm 用于观察归一化的行并行与内存访问参数。
这些是代表性工作负载选择，不是对两张卡实际瓶颈的测量结论。缩减 shape/dtype/case 数量，
保留 full 搜索域：GEMM 是截断的 seeded Grid Search，RMSNorm 的 Grid 可以遍历整个小搜索域。
RMSNorm 不足以代表大空间中的 BO 收敛，仅作小空间对照。

- 搜索 seeds 固定为 `[0, 1]`；输入 data_seed 使用 Workload 默认值 17，两个验证输入为 17、18。
- 每次候选 warmup=10、repeats=15、batch=16、validation_seeds=2。
- Grid、Random、multivariate TPE 使用同一搜索空间及相同 **attempt** 上限。
  失败、剪枝、超时、重复候选均消耗一次；Random/TPE 不保证相同 unique 配置数。
  不补跑失败候选以凑齐成功次数。方法顺序按 seed 打乱。
- 每个 worker timeout=60 秒；不设置 study `time_budget`，避免不同算法因时间截止而得到不同候选数。
  这是候选预算控制，不是收费上限，也不是保证完成时间。出现 timeout 时保留失败证据；
  若要调整 timeout，应对两卡所有方法统一协议并使用新目录，不能仅给失败方法额外预算。
- 所有候选仍为独立进程、冷编译，测量为 CUDA graph / 重用热 buffer；
  收敛速度包含失败成本，不能将 kernel latency 与调优总时间混为一谈。

| 阶段 | Studies | 最大 attempts |
|---|---:|---:|
| 一张卡 cold pilot | 12 | 216 |
| 两张卡分别 cold pilot | 24 | 432 |
| 目标卡 cold + warm pilot | 16 | 288 |
| 3090 cold + 5090 cold/warm（推荐迁移路线） | 28 | 504 |

单卡 cold 的 worker timeout 之和为 12,960 秒（3.6 小时），目标 cold/warm 为 17,280 秒（4.8 小时）。
这只是极端情况下的 worker 时间额度，**不是预期时长或整套运行上限**；GPU 测试、数据库、
报告与归档有额外开销。完成后需人工停止收费资源；脚本不会停租。本次开发不创建任何实例。
中断后的 study 可能从头重跑，因此恢复会增加实际累计计算量；上表不包含被归档的旧尝试。

## 先 dry-run，再由操作者启动

在两台独立 Linux 主机上使用相同的、含本 PR 的源码 commit 和依赖环境。
现有 2026-10-28 GHCR digest **不包含 pilot 及新增校验/计时**，不能直接把旧镜像的
`BO_SUITE` 改为 pilot。可使用此 commit 的 checkout 和现有 bootstrap 安装流程，
或另行审核并构建/发布新镜像；此 PR 不发布镜像。GPU preflight 继续要求
torch 2.9.0+cu128、CUDA runtime 12.8、Triton 3.5.0，实际兼容性仍待两卡 smoke 验证。

以下命令从仓库根目录运行。dry-run 不导入 torch/triton，不创建实验目录，也不检查真实 CUDA。

```bash
python -m runpod.launch pilot --dry-run --output /workspace/results/3090-pilot
python -m runpod.launch runpod/configs/pilot.json --dry-run --output /workspace/results/5090-pilot

# 仅规划后续迁移；源结果内容、精度及目标 GPU 合法性会在真实执行 preflight 验证。
python -m runpod.launch pilot --dry-run \
  --source-run /workspace/imported/3090-pilot --source-seed 0 --top-k 3 \
  --output /workspace/results/5090-pilot-transfer
```

输出包括 studies、max_attempts、各方法总预算、每个 study 的候选上限/搜索空间大小、
绝对输出目录、归档及计时文件路径。拒绝无效参数、超过目标预算的 top-k、占用的输出路径、
文件型父路径以及 source/output 的相等或嵌套关系。dry-run 不证明目录最终可写、磁盘足够或 GPU 可用。

以下是**后续真实运行命令，本次未执行**。先单独通过两卡的 smoke，再运行 pilot；
pilot 自身也保留完整 19 项 GPU 测试门禁，不会跳过精度验证。

```bash
# RTX 3090 主机，已安装同版本环境后
bash runpod/run.sh pilot --device 0 --output /workspace/results/3090-pilot

# RTX 5090 主机：独立 cold pilot，可选路线
bash runpod/run.sh pilot --device 0 --output /workspace/results/5090-pilot

# 或：将 3090 的完整结果包复制并解压到另一台主机后，一次运行 cold + warm
bash runpod/run.sh pilot --device 0 \
  --source-run /workspace/imported/3090-pilot --source-seed 0 --top-k 3 \
  --output /workspace/results/5090-pilot-transfer

# 同一环境/设备/配置恢复时，原样保留上述参数并添加 --resume
```

迁移只从预先固定的 source seed=0 选取最多 3 个不同的合法参数组，不按目标结果挑 source seed。
它们计入目标 24/12 次预算，并在 5090 重新验证、重新测量；源 latency 不进入目标 study。
少于 3 个合法成功候选时实际入队数写入 `metadata.transfer.enqueued`；一个都没有则失败。
固定一个源 seed、两个目标 seed 的 pilot 只能探索迁移方向，完整实验需要更多源/目标种子。

`--source-run` 模式会重跑目标三种 cold 方法；不能向已有 cold 输出追加 warm study。
如果已经做了独立 5090 cold pilot，再运行 cold/warm，会额外增加 288 次尝试（总计 720）。
分析时只使用匹配的 cold/warm 运行，避免重复 method/seed；迁移比较应在 5090 内部进行，
不要拿 3090 与 5090 的原始毫秒值直接推断搜索算法优势。

## 证据与读数

| 需要记录的信息 | 现有或新增产物 |
|---|---|
| GPU UUID、型号、compute capability、显存 | `manifest.json.hardware`、各 study `metadata.hardware` |
| 实际软件/驱动、Python、代码版本 | `manifest.json`、`environment.json`（含 nvidia-smi）、`pip-freeze.txt` |
| 精度与执行状态 | `gpu-tests.xml/log`；逐 trial `attrs.measurement.validation` 的误差、rtol/atol；失败日志 |
| 候选尝试、成功、失败、剪枝、重复 | `result.json`、`trials.csv`、`summary.csv`；对照预算，不能只看 best |
| 收敛曲线 | `figures/<case>/<cohort>/convergence.pdf/png`，同时按 attempts 和调优 wall time |
| 每 study 总时间、到目标时间 | `result.json.elapsed_s`、`summary.csv.total_s/seconds_to_target` |
| **整套运行总时间（含归档）** | 输出目录旁的 `<run>.timing.json`；每次运行状态/UTC 时间/elapsed，以及累计 `total_active_wall_s` |
| 源参数及迁移来源 | `metadata.transfer`、manifest 的 source hash、trial `warm_start_rank` |

计时 sidecar 在归档完成后写出，避免修改已封存的 archive/checksums。复制结果时同时保留
`<run>.tar.gz` 和 `<run>.timing.json`。累计时间包括失败与恢复的执行时间，不含两次恢复之间的闲置时间、
安装环境时间或最后一次 timing 写盘；突然 SIGKILL/断电可能缺少该次总耗时，应标记不完整，不能填零。
它不是云计费时长。所有产物保留实际测量值，不预填 GPU 性能数据。

两个 seed 的 median/IQR 仅为描述性读数，paired 表的 bootstrap CI 应为空。
RMSNorm 的 12 点 Grid 最优仅代表当前固定搜索域与测量协议；GEMM 的 best observed 不是全局最优。
目标阈值为该设备 cohort 内 observed best 的 1.05 倍，存在选择偏差；未命中目标的数据保留为删失结果。
先检查失败率、精度、噪声与两种子的方向是否一致，再决定是否扩大预算。

正式研究使用 performance 的更多 seeds、dtype 和 workloads，并事先确定比较指标、预算及排除规则。
五个 seeds 是扩展实验起点，不自动保证统计说服力；需按 pilot 方差、重复测量稳定性和预期效应
决定样本量，确认结论对输入 shape、候选预算、source seed 及 GPU 状态的稳健性。

## CPU 验证

```bash
python -m pytest -q -m 'not gpu and not compile'
python -m ruff check .
python -m ruff format --check .
```

回归覆盖两卡静态合法域、计划规模、无 CUDA 的 dry-run、路径冲突、非法数值、真实 sampler
在 CPU 合成目标上的公平计数，以及成功/失败/恢复/归档的计时。合成目标不进入用户实验目录。
