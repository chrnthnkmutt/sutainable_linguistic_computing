"""Intel / AMD CPU: utilisation via psutil, energy via RAPL
(Linux powercap sysfs, or the Windows Energy Meter Interface on Windows).

AMD Zen CPUs expose the same RAPL interface under /sys/class/powercap/intel-rapl (amd_energy /
intel_rapl driver since Linux 5.8), so the monitor works for both; the key stays "intel_cpu" so
existing --hardware commands keep working.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from . import windows_emi
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
    label = "Intel/AMD CPU"
    kind = "cpu"

    @classmethod
    def detect(cls) -> Tuple[bool, str]:
        if is_apple_silicon():
            return False, "this is an Apple Silicon Mac"
        brand = cpu_brand()
        if not any(v in brand.lower() for v in ("intel", "amd")):
            return False, f"CPU is neither Intel nor AMD ({brand})"
        if sys.platform == "win32" and not windows_emi.package_instances():
            return True, f"{brand} (usage only: no Windows Energy Meter for this CPU)"
        if not (sys.platform.startswith("linux") or sys.platform == "win32"):
            return True, f"{brand} (usage only: no RAPL access on this OS)"
        return True, brand

    @classmethod
    def hardware_name(cls) -> str:
        return cpu_brand()

    def _begin(self) -> None:
        self._read_uj: Optional[Callable[[], float]] = None  # cumulative package energy (µJ)
        self._dram: Optional[_RaplCounter] = None
        self._emi = None
        self._last_uj = 0.0
        self._last_t = time.perf_counter()
        psutil.cpu_percent(interval=None)  # prime the utilisation counter
        if sys.platform == "win32":
            self._begin_windows()
            return
        if not sys.platform.startswith("linux"):
            self._note("CPU energy counters (RAPL) are not accessible on this OS: CPU power/energy = N/A")
            return
        packages, dram = _find_domains()
        if not packages:
            self._note("No RAPL domains found under /sys/class/powercap: CPU power/energy = N/A")
            return
        try:
            pkg = _RaplCounter(packages)
            self._read_uj = lambda: float(pkg.read())
            self._dram = _RaplCounter(dram) if dram else None
        except PermissionError:
            self._note("RAPL counters are not readable: run with sudo (root) for CPU power/energy")
        except OSError as exc:
            self._note(f"RAPL unavailable ({exc}): CPU power/energy = N/A")

    def _begin_windows(self) -> None:
        """RAPL through the Windows Energy Meter Interface (package domain; no DRAM domain)."""
        instances = windows_emi.package_instances()
        if not instances:
            self._note("No Windows Energy Meter (EMI) for this CPU: CPU power/energy = N/A")
            return
        try:
            self._emi = windows_emi.EmiEnergyCounter(instances)
            self._read_uj = lambda: self._emi.read_j() * 1e6
            self.extra["cpu_energy_source"] = "Windows EMI " + ",".join(instances)
        except OSError as exc:
            self._note(f"Windows Energy Meter unavailable ({exc}): CPU power/energy = N/A")

    def _sample(self) -> Optional[Dict[str, Any]]:
        now = time.perf_counter()
        power = None
        if self._read_uj is not None:
            uj = self._read_uj()
            if self._dram is not None:
                self._dram.read()
            dt = now - self._last_t
            if dt > 0:
                power = (uj - self._last_uj) / 1e6 / dt
            self._last_uj, self._last_t = uj, now
        return {"usage": psutil.cpu_percent(interval=None), "power": power}

    def _end(self) -> None:
        if self._read_uj is not None:
            self._energy_j = self._read_uj() / 1e6
        if self._dram is not None:
            self.extra["dram_j"] = self._dram.total_uj / 1e6
        if self._emi is not None:
            self._emi.close()
