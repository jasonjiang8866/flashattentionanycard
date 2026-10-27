"""One persistent launch for ragged experts with packed tokens and shared N/K.

Routing, sorting, activation, and scatter are deliberately outside this operator.
Empty experts are supported. Every output tile has exactly one writer.
"""

import triton
import triton.language as tl


@triton.jit
def _grouped(
    X,
    W,
    Y,
    OFFSETS,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    CTAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    base = tl.full((), 0, tl.int64)
    ntiles = tl.cdiv(N, BLOCK_N)
    for expert in range(E):
        start = tl.load(OFFSETS + expert)
        end = tl.load(OFFSETS + expert + 1)
        count = end - start
        tiles = tl.cdiv(count, BLOCK_M) * ntiles
        while tile < base + tiles:
            relative = tile - base
            rows = (relative // ntiles) * BLOCK_M + tl.arange(0, BLOCK_M)
            cols = (relative % ntiles) * BLOCK_N + tl.arange(0, BLOCK_N)
            ks = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for block in range(tl.cdiv(K, BLOCK_K)):
                kk = block * BLOCK_K + ks
                x = tl.load(
                    X + (start + rows[:, None]) * K + kk[None, :],
                    (rows[:, None] < count) & (kk[None, :] < K),
                    other=0,
                )
                w = tl.load(
                    W + expert * K * N + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N),
                    other=0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                Y + (start + rows[:, None]) * N + cols[None, :],
                acc,
                (rows[:, None] < count) & (cols[None, :] < N),
            )
            tile += CTAS
        base += tiles


def launch(x, weights, offsets, out, params, sm_count):
    p = dict(params)
    ctas = sm_count * p.pop("CTA_MULTIPLIER")
    e, k, n = weights.shape
    return _grouped[(ctas,)](x, weights, out, offsets, n, k, e, ctas, **p)
