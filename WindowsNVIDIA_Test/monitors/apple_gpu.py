"""Apple Silicon GPU: utilisation from ioreg (no sudo) + IOReport energy via zeus-apple-silicon."""
from __future__ import annotations

import json
import re
import subprocess
from functools import lru_cache
from typing import Optional, Tuple

from .apple_common import AppleEnergyBase, apple_detect


@lru_cache(maxsize=1)
def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        gpu = json.loads(out)["SPDisplaysDataType"][0]
        name = gpu.get("sppci_model", "Apple GPU")
        cores = gpu.get("sppci_cores")
        return f"{name} GPU ({cores} cores)" if cores else f"{name} GPU"
    except Exception:
        return "Apple Silicon GPU"


def gpu_utilization() -> Optional[float]:
    """Device utilisation % from the IOAccelerator PerformanceStatistics."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception:
        return None
    m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
    return float(m.group(1)) if m else None


class AppleGPUMonitor(AppleEnergyBase):
    key = "apple_gpu"
    label = "Apple Silicon GPU"
    kind = "gpu"
    energy_field = "gpu_mj"

    @classmethod
    def detect(cls) -> Tuple[bool, str]:
        ok, why = apple_detect()
        return (True, gpu_name()) if ok else (False, why)

    @classmethod
    def hardware_name(cls) -> str:
        return gpu_name()

    def _begin(self) -> None:
        super()._begin()
        if gpu_utilization() is None:
            self._note("Apple GPU utilisation % not readable via ioreg on this system: GPU usage = N/A")

    def _usage(self) -> Optional[float]:
        return gpu_utilization()
