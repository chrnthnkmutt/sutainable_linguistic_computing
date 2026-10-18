"""Shared Apple Silicon energy plumbing (zeus-apple-silicon / IOReport, no sudo needed)."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from .base import BaseMonitor, is_apple_silicon

_energy_monitor = None


def get_energy_monitor():
    global _energy_monitor
    if _energy_monitor is None:
        from zeus_apple_silicon import AppleEnergyMonitor

        _energy_monitor = AppleEnergyMonitor()
    return _energy_monitor


def begin_window(mon, label: str) -> None:
    try:
        mon.begin_window(label, restart=True)
    except TypeError:
        mon.begin_window(label)


def apple_detect() -> Tuple[bool, str]:
    if not is_apple_silicon():
        return False, "not an Apple Silicon Mac"
    try:
        import zeus_apple_silicon  # noqa: F401
    except ImportError:
        return False, "zeus-apple-silicon is not installed (pip install zeus-apple-silicon)"
    return True, ""


class AppleEnergyBase(BaseMonitor):
    """Runs one long window (total energy) plus a rolling short window (per-sample power)."""

    energy_field = ""  # AppleEnergyMetrics attribute, e.g. "cpu_total_mj"

    def _usage(self) -> Optional[float]:
        raise NotImplementedError

    def _begin(self) -> None:
        self._mon = get_energy_monitor()
        self._run_label = f"{self.key}_run"
        self._tick_label = f"{self.key}_tick"
        begin_window(self._mon, self._run_label)
        begin_window(self._mon, self._tick_label)
        self._last_t = time.perf_counter()

    def _sample(self) -> Optional[Dict[str, Any]]:
        now = time.perf_counter()
        m = self._mon.end_window(self._tick_label)
        begin_window(self._mon, self._tick_label)
        dt, self._last_t = now - self._last_t, now
        mj = getattr(m, self.energy_field, None)
        power = mj / 1000.0 / dt if (mj is not None and dt > 0) else None
        return {"usage": self._usage(), "power": power}

    def _end(self) -> None:
        m = self._mon.end_window(self._run_label)
        try:
            self._mon.end_window(self._tick_label)
        except Exception:
            pass
        mj = getattr(m, self.energy_field, None)
        if mj is None:
            self._note(f"IOReport did not expose '{self.energy_field}' on this chip: energy = N/A")
        else:
            self._energy_j = mj / 1000.0
        for name, field in (("dram_j", "dram_mj"), ("ane_j", "ane_mj"), ("gpu_sram_j", "gpu_sram_mj")):
            val = getattr(m, field, None)
            if val is not None:
                self.extra[name] = val / 1000.0
