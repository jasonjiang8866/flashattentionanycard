"""Compile masked/ragged kernels to Ampere and Blackwell without a CUDA device.

This verifies code generation, not numerical correctness or runtime resource fit.
"""

import pytest

from bo_kernel_tuner.configs.search_spaces import grid_configs, search_space


@pytest.mark.compile
@pytest.mark.parametrize("architecture", [80, 86, 120])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kernel", ["gemm", "rmsnorm", "moe_grouped_gemm"])
def test_offline_compile(kernel, dtype, architecture):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from bo_kernel_tuner.kernels.gemm import _gemm
    from bo_kernel_tuner.kernels.moe_grouped_gemm import _grouped
    from bo_kernel_tuner.kernels.rmsnorm import _rmsnorm

    for params in grid_configs(search_space(kernel, "smoke")):
        p = dict(params)
        options = {"num_warps": p.pop("num_warps"), "num_stages": p.pop("num_stages", 1)}
        if kernel == "gemm":
            fn = _gemm
            pointers = {name: f"*{dtype}" for name in ("A", "B", "C")}
            constants = dict(M=37, N=77, K=93, **p)
        elif kernel == "rmsnorm":
            fn = _rmsnorm
            pointers = {name: f"*{dtype}" for name in ("X", "W", "Y")}
            constants = dict(M=37, N=77, EPS=1e-6, BLOCK=128, **p)
        else:
            fn = _grouped
            pointers = {name: f"*{dtype}" for name in ("X", "W", "Y")}
            pointers["OFFSETS"] = "*i64"
            constants = dict(N=77, K=93, E=5, CTAS=82 * p.pop("CTA_MULTIPLIER"), **p)
        compiled = triton.compile(
            ASTSource(fn, signature=pointers, constexprs=constants),
            target=GPUTarget("cuda", architecture, 32),
            options=options,
        )
        assert compiled.asm["cubin"]
        assert compiled.metadata.shared >= 0
