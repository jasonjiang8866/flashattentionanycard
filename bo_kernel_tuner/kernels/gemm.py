"""Contiguous FP16/BF16 GEMM, FP32 accumulation, masked edge tiles."""

import triton
import triton.language as tl


@triton.jit
def _gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm, pn = first_m + local % group_m, local // group_m
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        kk = block * BLOCK_K + ks
        a = tl.load(
            A + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), other=0
        )
        b = tl.load(
            B + kk[:, None] * N + cols[None, :], (kk[:, None] < K) & (cols[None, :] < N), other=0
        )
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def launch(a, b, out, params):
    m, k = a.shape
    n = b.shape[1]
    return _gemm[(triton.cdiv(m, params["BLOCK_M"]) * triton.cdiv(n, params["BLOCK_N"]),)](
        a, b, out, m, n, k, **params
    )
