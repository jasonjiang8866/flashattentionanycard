# RunPod 实验入口

提供现成 Pod 的 bootstrap 和可构建的 Docker 环境。默认只运行 smoke；本仓库没有自动创建/租用 Pod，也没有将 CPU 检查标成真实 GPU 验证。

## 1. 最快入口：在现有 RunPod GPU Pod 中执行

选择 Linux x86_64、NVIDIA Ampere 或更新 GPU，使用能够打开终端的 RunPod PyTorch 模板。
驱动必须支持所选 GPU 与 CUDA 12.8 wheel。建议容器磁盘 30 GB、可写 `/workspace` 至少 20 GB；
bootstrap 的独立 Python 环境也默认保存在 `/workspace`。

```bash
cd /workspace
git clone --branch feat/runpod-experiments \
  https://github.com/jasonjiang8866/flashattentionanycard.git
cd flashattentionanycard
bash runpod/bootstrap.sh smoke \
  --output /workspace/bo-kernel-tuner/results/smoke-01
```

这条命令会：

1. 创建受脚本管理的独立 venv，安装 PyTorch **2.9.0+cu128**、Triton **3.5.0** 及固定版本的实验依赖。
2. 检查 CUDA、设备属性与环境版本；在所选 GPU 上运行全部 19 项真实 GPU 测试，任何 skip/failure/error 都阻止继续。
3. 对三个算子的 FP16/BF16 运行 Grid、Random、TPE 调优。
4. 校验结果完整性，生成统计图、CSV、校验和及 `.tar.gz` 结果包。

bootstrap 不修改模板的系统 Python，不会覆盖已有的非本项目 venv。
依赖安装完成后的重复运行使用 `run.sh`，无需再安装：

```bash
bash runpod/run.sh smoke --device 0 \
  --output /workspace/bo-kernel-tuner/results/smoke-02

# 只查看实验计划：无 CUDA 访问、无实验目录写入。
bash runpod/run.sh performance --dry-run

# 明确选择完整实验：
bash runpod/run.sh performance --device 0 \
  --output /workspace/bo-kernel-tuner/results/ampere-full
```

| 配置 | 算子与 dtype | seeds | studies | 最大候选评估次数 |
|---|---|---:|---:|---:|
| `smoke` | 3 算子 × FP16/BF16 | 1 | 18 | 36 |
| `performance` | 3 算子 × FP16/BF16 | 5 | 90 | 3,360 |
| `performance` 加迁移 | 上述配置 + warm TPE | 5 | 120 | 4,480 |

smoke 验证执行链路，不能证明 TPE 的收敛优势。performance 包含每个 GEMM/MoE study 50 attempts、RMSNorm 12 attempts；
每次都独立启动进程并冷编译，完整实验可能耗时数小时，具体以 GPU、编译和测量时间为准。
自定义预算、shape、seed 或测量次数时，复制 `configs/performance.json` 为自己的 JSON，作为第一个参数传入。

长时间运行可使用模板已有的 tmux，或：

```bash
nohup bash runpod/run.sh performance \
  --output /workspace/bo-kernel-tuner/results/ampere-full \
  > /workspace/bo-kernel-tuner/ampere-full-console.log 2>&1 &
```

## 2. 跨 GPU 迁移

源 Pod 运行 `performance` 后，保留整个结果目录或解压结果包，复制到目标 Pod 的 `/workspace`。
使用同一代码版本、同一配置和同一环境；两个 GPU 不需要在同一主机。

```bash
# 在 Blackwell Pod 上，先用 bootstrap 安装相同环境并通过 smoke。
# 此处源目录必须含 cases/<case-id>/tpe-seed-0/{result.json,receipt.json,...}。
bash runpod/run.sh performance --device 0 \
  --source-run /workspace/bo-kernel-tuner/results/ampere-full \
  --source-seed 0 --top-k 5 \
  --output /workspace/bo-kernel-tuner/results/blackwell-transfer
```

每个 case 都必须有匹配、完整、通过校验和检查的源 TPE study。缺失来源或测量协议不匹配时直接失败，不静默退回冷启动。
目标端同时运行三种冷启动方法及 warm TPE；候选只迁移参数，全部在目标 GPU 重测。
测试 smoke 的迁移链路时须用 `--top-k 1`，因为 smoke 的单 study 预算只有 2。

## 3. 恢复与结果完整性

```bash
bash runpod/run.sh performance \
  --output /workspace/bo-kernel-tuner/results/ampere-full --resume
```

`--resume` 会核对配置、代码指纹、Python/依赖版本、硬件身份与来源文件哈希。
完整 study 必须同时具备正确的 metadata、有效测量样本、精度验证记录、best trial、CSV、SQLite 及 receipt 校验和才会跳过。
完成标记不能替代文件检查。

恢复粒度为 **study**：中断/失败 study 的旧数据移动到 `incomplete/`，该 study 从头重新运行。
当前不恢复 Optuna sampler 的随机数状态，所以不会把新采样接在旧 trial 后伪装成连续实验。
正式论文分析应保留重跑记录，并避免通过反复重试筛掉失败 run。
若已有完成结果被修改或缺少文件，恢复会报错；请检查原因或使用新目录。

SIGTERM/Ctrl-C 会尽可能保存状态与压缩包；突然断电或 SIGKILL 可能只留下逐 trial 的检查点。
没有 manifest 的失败目录须使用新输出目录重试。

| 产物 | 用途 |
|---|---|
| `status.json` | phase、总体状态、GPU 测试与完整实验验证状态 |
| `manifest.json` / `environment.json` / `pip-freeze.txt` | 配置、代码、设备和实际依赖环境 |
| `gpu-tests.xml` / `gpu-tests.log` | 真实 GPU 测试的机器可读结果及日志 |
| `cases/<case>/<method>-seed-N/` | Optuna SQLite、JSON、CSV、worker 日志与 receipt |
| `figures/<case>/` | 收敛曲线、耗时图、参数热力图、统计表 |
| `checksums.json` | 输出文件 SHA-256 索引 |
| 同级 `<run-name>.tar.gz` | 完整结果包；失败时也尽可能生成 |

只有 GPU 测试无跳过、每个计划 study 通过完整性检查、图表和统计表齐全时，
`status=complete` 且 `gpu_validation_passed=true`。恢复过程中该标记保持 false。

`/workspace` 是否在终止 Pod 后保留取决于挂载的存储类型：需要跨 Pod 保留时使用 Network Volume，或先下载结果包。
普通 Pod volume 可随 Pod 终止被删除。脚本不自动停租；完成后请在 RunPod 控制台停止/终止 Pod，并先确认结果已保留。

## 4. Docker 环境

`runpod/Dockerfile` 使用 Python 3.12 和同一套固定直接依赖。CUDA 用户态库和 ptxas 来自 PyTorch/Triton wheels；
GPU 驱动由宿主机提供。镜像不安装宿主驱动，不包含模型下载，不需要 Hugging Face token。

在支持 Docker 的机器上构建并推送到自己的 registry：

```bash
docker build --platform linux/amd64 -f runpod/Dockerfile \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t YOUR_REGISTRY/bo-kernel-tuner:YOUR_TAG .
docker push YOUR_REGISTRY/bo-kernel-tuner:YOUR_TAG
```

RunPod 自定义模板：

- Container Image：上面已推送的镜像，正式实验优先使用 registry 返回的 digest。
- Volume Mount Path：`/workspace`。
- Start Command：留空使用镜像 CMD，或设为 `bash /opt/bo-kernel-tuner/runpod/entrypoint.sh`。
- 默认环境：`BO_SUITE=smoke`、`BO_DEVICE=0`；输出为 `/workspace/bo-kernel-tuner/results/current`。
- 此镜像为批处理环境，未配置 SSH/Jupyter 服务或公开端口。需要交互式开发时优先使用第 1 节的现成 RunPod 模板。

Pod 启动后自动跑一次实验，随后保持容器运行以便取回文件。发现已有 `status.json` 或 manifest 时不会自动再跑；
使用新 `BO_OUTPUT_DIR` 启动新实验，或通过终端显式执行 `run.sh ... --resume`。
保持容器运行不代表停止 GPU 计费。

| 环境变量 | 默认值/用途 |
|---|---|
| `BO_SUITE` | `smoke`；也可为 `performance` 或 JSON 路径 |
| `BO_DEVICE` | `0`，当前可见设备中的逻辑编号 |
| `BO_WORKSPACE` | `/workspace/bo-kernel-tuner` |
| `BO_OUTPUT_DIR` | 手动运行自动生成唯一目录；镜像启动固定为 `results/current` |
| `BO_SOURCE_DIR` / `BO_TOP_K` | 可选迁移来源完整目录 / 默认 5 |
| `BO_VENV_DIR` | bootstrap 的自定义 venv 位置 |
| `BO_PYTHON` | 运行脚本使用的 Python；Docker 中为 `/opt/venv/bin/python` |

GitHub Actions 的 **RunPod image and offline validation** 工作流会构建镜像并在容器内执行离线测试。
手动执行该 workflow 并勾选 `publish` 才推送 GHCR，镜像名为
`ghcr.io/jasonjiang8866/flashattentionanycard-runpod:sha-<完整commit SHA>`。
确认发布成功后再填入 RunPod；GHCR package 若为私有，需配置 registry 凭据或自行设为公开。
CI 的 Docker 构建/离线编译通过仍不等于真实 GPU 实验已通过。

## 官方参考

- [RunPod storage types](https://docs.runpod.io/pods/storage/types)
- [RunPod custom templates](https://docs.runpod.io/pods/templates/create-custom-template)
- [PyTorch previous-version CUDA wheels](https://pytorch.org/get-started/previous-versions/)
