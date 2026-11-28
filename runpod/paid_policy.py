"""Immutable limits for a single reviewed paid experiment; no runtime overrides."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PaidPolicy:
    suite: str
    authorization: str
    expires: int
    approved_usd: float
    prior_usd: float
    unallocated_usd: float
    total_seconds: int
    prior_seconds: int
    max_work_seconds: int
    cleanup_seconds: int
    transfer_seconds: int
    min_work_seconds: int
    reconciled_markers: tuple[tuple[str, int], ...] = ()

    @property
    def marker(self):
        return "runpod-authorization-" + self.authorization


# User approved US$10 / 500 minutes on 2026-11-28. Conservatively carry the
# historical $0.41 reserve forward instead of treating this as $10 extra.
# The 500-minute time allowance covers this pilot; the first attempt uses at most
# 120 minutes work + 10 cleanup. Every additional attempt requires reconciliation.
# This seven-day launch window expires well before the marker's 90-day retention.
PILOT = PaidPolicy(
    suite="pilot",
    authorization="faac-pilot-20261128-usd10-500m",
    expires=1796428800,  # 2026-12-05 00:00 UTC
    approved_usd=10,
    prior_usd=0.41,
    unallocated_usd=1,
    total_seconds=500 * 60,
    prior_seconds=0,
    max_work_seconds=120 * 60,
    cleanup_seconds=10 * 60,
    transfer_seconds=180,
    min_work_seconds=30 * 60,
)
