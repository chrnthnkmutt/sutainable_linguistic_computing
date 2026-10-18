"""Apple Silicon CPU: psutil utilisation + IOReport energy through zeus-apple-silicon."""
from __future__ import annotations

from typing import Optional, Tuple

import psutil

from .apple_common import AppleEnergyBase, apple_detect
from .base import cpu_brand


class AppleCPUMonitor(AppleEnergyBase):
    key = "apple_cpu"
    label = "Apple Silicon CPU"
    kind = "cpu"
    energy_field = "cpu_total_mj"

    @classmethod
    def detect(cls) -> Tuple[bool, str]:
        ok, why = apple_detect()
        return (True, cpu_brand()) if ok else (False, why)

    @classmethod
    def hardware_name(cls) -> str:
        return cpu_brand()

    def _begin(self) -> None:
        psutil.cpu_percent(interval=None)
        super()._begin()

    def _usage(self) -> Optional[float]:
        return psutil.cpu_percent(interval=None)
