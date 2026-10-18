"""Common machinery shared by every hardware monitor.

A monitor is started right before inference and stopped right after it.
While running, a background thread takes a sample every ``interval`` seconds.
Sub-classes only implement ``_begin``, ``_sample`` and ``_end``.
"""
from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import threading
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


@lru_cache(maxsize=1)
def cpu_brand() -> str:
    """Human readable CPU name (e.g. 'Intel(R) Core(TM) i7-12700H', 'Apple M2 Pro')."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    try:
        import cpuinfo  # py-cpuinfo

        brand = cpuinfo.get_cpu_info().get("brand_raw") or ""
        if brand:
            return brand
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mean_or_none(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


class BaseMonitor:
    key = "base"
    label = "Base"
    kind = "cpu"  # "cpu" | "gpu" | "mem"

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.extra: Dict[str, Any] = {}
        self._energy_j: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._t0 = 0.0
        self._t1 = 0.0

    # ---- discovery -------------------------------------------------------
    @classmethod
    def detect(cls) -> Tuple[bool, str]:
        """Return (available, info). info = hardware name, or the reason it is unavailable."""
        return True, cls.label

    @classmethod
    def hardware_name(cls) -> str:
        return cls.label

    # ---- hooks for sub-classes ------------------------------------------
    def _begin(self) -> None: ...

    def _sample(self) -> Optional[Dict[str, Any]]:
        return None

    def _end(self) -> None: ...

    # ---- life-cycle ------------------------------------------------------
    def _note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    def start(self) -> None:
        self.samples, self.notes, self.extra = [], [], {}
        self._energy_j = None
        self._stop_evt.clear()
        try:
            self._begin()
        except Exception as exc:  # never let a monitor crash the experiment
            self._note(f"start failed: {exc}")
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"mon-{self.key}")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_evt.wait(self.interval):
            self._take_sample()

    def _take_sample(self) -> None:
        try:
            s = self._sample()
        except Exception as exc:
            self._note(f"sampling failed: {exc}")
            return
        if s is not None:
            s["t"] = round(time.perf_counter() - self._t0, 3)
            self.samples.append(s)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._take_sample()  # final sample so short runs still have data
        self._t1 = time.perf_counter()
        try:
            self._end()
        except Exception as exc:
            self._note(f"finalise failed: {exc}")

    # ---- output ----------------------------------------------------------
    def result(self) -> Dict[str, Any]:
        duration = max(self._t1 - self._t0, 1e-9)
        usage = [s["usage"] for s in self.samples if s.get("usage") is not None]
        powers = [s["power"] for s in self.samples if s.get("power") is not None]
        energy = self._energy_j
        if energy is not None:
            power_avg: Optional[float] = energy / duration
        else:
            power_avg = mean_or_none(powers)
        return {
            "key": self.key,
            "kind": self.kind,
            "name": self.hardware_name(),
            "duration_s": duration,
            "usage_start": usage[0] if usage else None,
            "usage_end": usage[-1] if usage else None,
            "usage_avg": mean_or_none(usage),
            "usage_peak": max(usage) if usage else None,
            "power_avg_w": power_avg,
            "power_peak_w": max(powers) if powers else None,
            "energy_j": energy,
            "extra": dict(self.extra),
            "notes": list(self.notes),
            "samples": list(self.samples),
        }
