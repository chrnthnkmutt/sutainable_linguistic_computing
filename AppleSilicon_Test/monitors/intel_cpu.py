"""Intel CPU: utilisation via psutil, energy via Intel RAPL (Linux powercap sysfs)."""
from __future__ import annotations

import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from .base import BaseMonitor, cpu_brand, is_apple_silicon

RAPL_ROOT = "/sys/class/powercap"


def _read(path: str) -> str:
    with open(path) as fh:
        return fh.read().strip()


class _RaplCounter:
    """Cumulative energy (µJ) over several RAPL domains, handling counter wrap-around."""

    def __init__(self, paths: List[str]):
        self.paths = paths
        self.max_range = {p: int(_read(p + "/max_energy_range_uj")) for p in paths}
        self.last = {p: int(_read(p + "/energy_uj")) for p in paths}
        self.total_uj = 0

    def read(self) -> int:
        for p in self.paths:
            cur = int(_read(p + "/energy_uj"))
            prev = self.last[p]
            self.total_uj += cur - prev if cur >= prev else cur + self.max_range[p] - prev
            self.last[p] = cur
        return self.total_uj


def _find_domains() -> Tuple[List[str], List[str]]:
    packages, dram = [], []
    for path in sorted(glob.glob(f"{RAPL_ROOT}/intel-rapl:*")):
        if os.path.basename(path).count(":") != 1:
            continue
        packages.append(path)
        for sub in sorted(glob.glob(path + "/intel-rapl:*:*")):
            try:
                if _read(sub + "/name") == "dram":
                    dram.append(sub)
            except OSError:
                pass
    return packages, dram


class IntelCPUMonitor(BaseMonitor):
    key = "intel_cpu"
    label = "Intel CPU"
    kind = "cpu"

    @classmethod
    def detect(cls) -> Tuple[bool, str]:
        if is_apple_silicon():
            return False, "this is an Apple Silicon Mac"
        brand = cpu_brand()
        if "intel" not in brand.lower():
            return False, f"CPU is not Intel ({brand})"
        return True, brand

    @classmethod
    def hardware_name(cls) -> str:
        return cpu_brand()

    def _begin(self) -> None:
        self._pkg: Optional[_RaplCounter] = None
        self._dram: Optional[_RaplCounter] = None
        self._last_uj = 0
        self._last_t = time.perf_counter()
        psutil.cpu_percent(interval=None)  # prime the utilisation counter
        if not sys.platform.startswith("linux"):
            self._note("Intel energy counters (RAPL) are only supported on Linux: CPU power/energy = N/A")
            return
        packages, dram = _find_domains()
        if not packages:
            self._note("No RAPL domains found under /sys/class/powercap: CPU power/energy = N/A")
            return
        try:
            self._pkg = _RaplCounter(packages)
            self._dram = _RaplCounter(dram) if dram else None
        except PermissionError:
            self._note("RAPL counters are not readable: run with sudo (root) for CPU power/energy")
        except OSError as exc:
            self._note(f"RAPL unavailable ({exc}): CPU power/energy = N/A")

    def _sample(self) -> Optional[Dict[str, Any]]:
        now = time.perf_counter()
        power = None
        if self._pkg is not None:
            uj = self._pkg.read()
            if self._dram is not None:
                self._dram.read()
            dt = now - self._last_t
            if dt > 0:
                power = (uj - self._last_uj) / 1e6 / dt
            self._last_uj, self._last_t = uj, now
        return {"usage": psutil.cpu_percent(interval=None), "power": power}

    def _end(self) -> None:
        if self._pkg is not None:
            self._energy_j = self._pkg.total_uj / 1e6
        if self._dram is not None:
            self.extra["dram_j"] = self._dram.total_uj / 1e6
