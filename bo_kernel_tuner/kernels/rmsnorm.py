"""Fused RMS reduction, normalization, and weight multiplication."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_CTA + tl.arange(0, ROWS_PER_CTA)
    cols = tl.arange(0, BLOCK)
    x = tl.load(
        X + rows[:, None] * N + cols[None, :], (rows[:, None] < M) & (cols[None, :] < N), other=0
    ).to(tl.float32)
    weight = tl.load(W + cols, cols < N, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=1) / N + EPS)
    y = x * scale[:, None] * weight[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], y, (rows[:, None] < M) & (cols[None, :] < N))


def launch(x, weight, out, params, epsilon=1e-6):
    m, n = x.shape
    return _rmsnorm[(triton.cdiv(m, params["ROWS_PER_CTA"]),)](
        x, weight, out, m, n, epsilon, triton.next_power_of_2(n), **params
    )
