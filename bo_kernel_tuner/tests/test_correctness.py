"""CPU legality tests plus real CUDA correctness tests (never mocked as GPU passes)."""

import pytest

from bo_kernel_tuner.configs.search_spaces import Workload, grid_configs, grid_size, search_space
from bo_kernel_tuner.tuner.hardware import Hardware, prune_reason, resource_estimate


@pytest.fixture
def hardware():
    return Hardware("test-device", (8, 6), 82, 24 * 1024**3, 99 * 1024)


@pytest.mark.parametrize("kernel", ["gemm", "rmsnorm", "moe_grouped_gemm"])
def test_shared_search_domain(kernel):
    space = search_space(kernel)
    configs = list(grid_configs(space))
    assert len(configs) == grid_size(space)
    assert len({tuple(p.items()) for p in configs}) == len(configs)


def test_hard_legality_and_heuristics_are_distinct(hardware):
    p = next(grid_configs(search_space("gemm")))
    p.update(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=4)
    assert resource_estimate("gemm", p)["shared_bytes_estimate"] > hardware.shared_memory_per_block
    assert prune_reason(hardware, "gemm", p) is None
    assert (
        prune_reason(hardware, "gemm", p, heuristic_smem=True) == "heuristic_shared_memory_estimate"
    )
    p["num_warps"] = 64
    assert prune_reason(hardware, "gemm", p) == "illegal_threads_per_block"


def test_rmsnorm_limit(hardware):
    assert prune_reason(hardware, "rmsnorm", {"ROWS_PER_CTA": 4, "num_warps": 4}, 32769)
    assert prune_reason(hardware, "rmsnorm", {"ROWS_PER_CTA": 1, "num_warps": 4}, 4097) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 0},
        {"dtype": "fp32"},
        {"kernel": "bad"},
        {"expert_tokens": (0, 0)},
        {"expert_tokens": (-1, 2)},
        {"epsilon": float("nan")},
        {"data_seed": -1},
    ],
)
def test_invalid_workload(kwargs):
    with pytest.raises(ValueError):
        Workload(**kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kernel", ["gemm", "rmsnorm", "moe_grouped_gemm"])
@pytest.mark.parametrize("shape", [(64, 128, 64), (37, 77, 93), (1, 1, 1)])
def test_gpu_outputs(kernel, dtype, shape):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if (
        not torch.cuda.is_available()
        or torch.version.hip
        or torch.cuda.get_device_capability()[0] < 8
    ):
        pytest.skip("NVIDIA Ampere+ CUDA device required")
    from bo_kernel_tuner.tuner.worker import check_output, prepare

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hardware = Hardware.detect(0)
    m, n, k = shape
    w = Workload(kernel, m, n, k, dtype, (0, 1, 33, 0, 9))
    # Includes masked tails, zero experts, grouped M boundaries, and persistent scheduling.
    for p in grid_configs(search_space(kernel, "smoke")):
        for seed in (0, 17):
            call, out, reference = prepare(w, torch.device("cuda:0"), seed, hardware.sm_count)
            out.fill_(float("nan"))
            call(p)
            torch.cuda.synchronize()
            check_output(out, reference, dtype)


@pytest.mark.gpu
def test_cuda_measurement():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from bo_kernel_tuner.tuner.worker import Measurement, measure

    x = torch.ones(1024, device="cuda")
    result = measure(lambda: x.add_(1), Measurement(2, 3, 2, 1))
    assert len(result["samples_ms"]) == 3
    assert result["latency_ms"] > 0
