"""Hard legality checks are separated from nonbinding resource estimates."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    capability: tuple[int, int]
    sm_count: int
    total_memory: int
    shared_memory_per_block: int
    max_threads_per_block: int = 1024
    warp_size: int = 32
    uuid: str = "unknown"
    device: int = 0

    def to_dict(self):
        data = asdict(self)
        data["capability"] = list(self.capability)
        return data

    @classmethod
    def detect(cls, device=0):
        import torch

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("This initial implementation requires NVIDIA CUDA (Ampere+)")
        p = torch.cuda.get_device_properties(device)
        if p.major < 8:
            raise RuntimeError("Compute capability >= 8.0 required")
        return cls(
            p.name,
            (p.major, p.minor),
            p.multi_processor_count,
            p.total_memory,
            getattr(p, "shared_memory_per_block_optin", p.shared_memory_per_block),
            getattr(p, "max_threads_per_block", 1024),
            getattr(p, "warp_size", 32),
            str(getattr(p, "uuid", "unknown")),
            device,
        )


def resource_estimate(kernel, params, columns=0):
    if kernel == "rmsnorm":
        block = 1 << (columns - 1).bit_length()
        return {"logical_elements": block * params["ROWS_PER_CTA"], "shared_bytes_estimate": 0}
    return {
        "shared_bytes_estimate": 2
        * params["BLOCK_K"]
        * (params["BLOCK_M"] + params["BLOCK_N"])
        * params["num_stages"],
        "accumulator_bytes": 4 * params["BLOCK_M"] * params["BLOCK_N"],
    }


def prune_reason(hardware, kernel, params, columns=0, heuristic_smem=False):
    warps = params["num_warps"]
    if (
        warps not in (1, 2, 4, 8, 16, 32)
        or warps * hardware.warp_size > hardware.max_threads_per_block
    ):
        return "illegal_threads_per_block"
    # An implementation limit, not a claimed hardware register/occupancy bound.
    if kernel == "rmsnorm" and (1 << (columns - 1).bit_length()) * params["ROWS_PER_CTA"] > 65536:
        return "rmsnorm_implementation_limit"
    if (
        heuristic_smem
        and resource_estimate(kernel, params, columns)["shared_bytes_estimate"]
        > hardware.shared_memory_per_block
    ):
        return "heuristic_shared_memory_estimate"
    return None
