"""One categorical domain shared by Grid, Random, and multivariate TPE."""

from dataclasses import asdict, dataclass
from itertools import product
from math import prod

SPACE_VERSION = 1
KERNELS = ("gemm", "rmsnorm", "moe_grouped_gemm")


@dataclass(frozen=True)
class Workload:
    kernel: str = "gemm"
    m: int = 1024
    n: int = 1024
    k: int = 1024
    dtype: str = "fp16"
    expert_tokens: tuple[int, ...] = (64, 128, 256, 512)
    epsilon: float = 1e-6
    data_seed: int = 17

    def __post_init__(self):
        if self.kernel not in KERNELS or self.dtype not in ("fp16", "bf16"):
            raise ValueError("Unknown kernel or dtype")
        if min(self.m, self.n, self.k) <= 0 or not 0 < self.epsilon < 1:
            raise ValueError("Dimensions and epsilon must be positive; epsilon < 1")
        if not self.expert_tokens or min(self.expert_tokens) < 0 or sum(self.expert_tokens) <= 0:
            raise ValueError("Experts require nonnegative token counts and a positive total")
        if self.data_seed < 0:
            raise ValueError("data_seed must be nonnegative")

    def signature(self):
        """Only semantically relevant fields enter transfer compatibility."""
        d = asdict(self)
        if self.kernel == "rmsnorm":
            for key in ("k", "expert_tokens"):
                d.pop(key)
        else:
            d.pop("epsilon")
            if self.kernel == "gemm":
                d.pop("expert_tokens")
            else:
                d.pop("m")
                d["expert_tokens"] = list(self.expert_tokens)
        return d


def search_space(kernel: str, profile: str = "full") -> dict[str, list[int]]:
    if kernel not in KERNELS or profile not in ("full", "smoke"):
        raise ValueError("Unknown kernel or profile")
    if kernel == "rmsnorm":
        return {
            "ROWS_PER_CTA": [1, 2, 4, 8] if profile == "full" else [1, 2],
            "num_warps": [4, 8, 16] if profile == "full" else [4, 8],
        }
    space = {
        "BLOCK_M": [16, 32, 64, 128],
        "BLOCK_N": [32, 64, 128],
        "BLOCK_K": [32, 64],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4],
    }
    if profile == "smoke":
        space = {
            "BLOCK_M": [16, 32],
            "BLOCK_N": [32],
            "BLOCK_K": [32],
            "num_warps": [4],
            "num_stages": [2],
        }
    if kernel == "gemm":
        space["GROUP_M"] = [1, 4, 8] if profile == "full" else [1, 4]
    else:
        space["CTA_MULTIPLIER"] = [1, 2, 4] if profile == "full" else [1, 2]
    return space


def grid_size(space):
    return prod(len(v) for v in space.values())


def grid_configs(space):
    return (dict(zip(space, values)) for values in product(*space.values()))


def validate_params(params, space):
    return set(params) == set(space) and all(
        type(params[k]) is int and params[k] in choices for k, choices in space.items()
    )
