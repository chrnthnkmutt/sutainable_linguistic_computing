"""Always-on memory monitor (process RAM incl. child processes + system RAM)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import psutil

from .base import BaseMonitor


class MemoryMonitor(BaseMonitor):
    key = "memory"
    label = "Memory"
    kind = "mem"

    def __init__(self, interval: float = 0.5, watch_names=()):
        super().__init__(interval)
        # also count processes whose name contains one of these (e.g. "ollama" for an Ollama server)
        self.watch_names = tuple(n.lower() for n in watch_names)

    def _begin(self) -> None:
        self._proc = psutil.Process()

    def _rss_mb(self) -> float:
        total = self._proc.memory_info().rss
        counted = {self._proc.pid}
        try:
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                    counted.add(child.pid)
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        if self.watch_names:
            for p in psutil.process_iter(["name", "memory_info"]):
                try:
                    name = (p.info["name"] or "").lower()
                    mi = p.info["memory_info"]
                    if mi is not None and p.pid not in counted and any(w in name for w in self.watch_names):
                        total += mi.rss
                except psutil.Error:
                    pass
        return total / 1048576

    def _sample(self) -> Optional[Dict[str, Any]]:
        vm = psutil.virtual_memory()
        return {"rss_mb": self._rss_mb(), "sys_pct": vm.percent, "sys_used_mb": vm.used / 1048576}

    def result(self) -> Dict[str, Any]:
        rss = [s["rss_mb"] for s in self.samples]
        pct = [s["sys_pct"] for s in self.samples]
        used = [s["sys_used_mb"] for s in self.samples]
        return {
            "key": self.key, "kind": self.kind, "name": "Memory",
            "duration_s": max(self._t1 - self._t0, 1e-9),
            "rss_start_mb": rss[0] if rss else None,
            "rss_end_mb": rss[-1] if rss else None,
            "rss_peak_mb": max(rss) if rss else None,
            "sys_pct_start": pct[0] if pct else None,
            "sys_pct_end": pct[-1] if pct else None,
            "sys_used_peak_mb": max(used) if used else None,
            "notes": list(self.notes), "samples": list(self.samples),
        }
