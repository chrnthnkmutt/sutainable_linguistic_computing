"""Energy measurement on Zeus (ml.energy).

Energy timeline used by ``bench.py``: hardware counters are read
only at ``mark()`` (request boundaries) and a background thread builds a power timeline for
phase splits and peaks.

Domains (whatever Zeus finds on this machine):
  gpu_<i>            NVIDIA / AMD GPU   counter: get_total_energy_consumption (mJ)
                                        trace:   get_instant_power_usage (mW) — no observer effect
  cpu_<i>, dram_<i>  Intel / AMD RAPL   counter: get_total_energy_consumption (mJ); Zeus tracks
                                        wrap-around itself. Linux only (needs read access to
                                        /sys/class/powercap); unavailable on Windows.
  apple_<part>       Apple Silicon      counter: zeus-apple-silicon / IOReport (mJ), no sudo:
                                        cpu_total, gpu, gpu_sram, ane, dram
  cpu_pkg            Intel / AMD on     counter: Windows Energy Meter (RAPL via EMI, pWh), no admin;
                     Windows            used when Zeus has no CPU source (win_emi.py)

Why counters only at marks: polling NVML's energy counter itself raises GPU power (+51 W at
10 Hz on an idle RTX 5070 Ti), so the GPU trace uses instantaneous power
instead. CPU / SoC counters are sampled for their trace (no such effect known).
"""
from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

APPLE_PARTS = ("cpu_total", "gpu", "gpu_sram", "ane", "dram")


@dataclass
class Mark:
    t: float
    counters: List[float]  # cumulative joules per domain


class ZeusTimeline:
    def __init__(self, period: float = 0.05, gpu_indices: Optional[List[int]] = None,
                 log: Callable[[str], None] = print):
        from zeus.device import get_cpus, get_gpus, get_soc

        self.period = period
        self.domains: List[str] = []
        self._readers: List[Callable[[], List[float]]] = []  # each returns joules for its domains
        self._gpu_power: List[Optional[Callable[[], float]]] = []  # per domain, W (None = from counter)

        try:
            gpus = get_gpus()
            idx = gpu_indices if gpu_indices is not None else list(range(len(gpus)))
            for i in idx:
                if gpus.supports_get_total_energy_consumption(i):
                    self.domains.append(f"gpu_{i}")
                    self._readers.append(lambda i=i: [gpus.get_total_energy_consumption(i) / 1000.0])
                    self._gpu_power.append(lambda i=i: gpus.get_instant_power_usage(i) / 1000.0)
            log(f"[energy] zeus GPUs: {[d for d in self.domains if d.startswith('gpu_')]}")
        except Exception as e:
            log(f"[energy] zeus GPUs: unavailable ({type(e).__name__}: {e})")

        try:
            cpus = get_cpus()
            for i in range(len(cpus)):
                dram = cpus.supports_get_dram_energy_consumption(i)
                names = [f"cpu_{i}"] + ([f"dram_{i}"] if dram else [])

                def read(i=i, dram=dram):
                    m = cpus.get_total_energy_consumption(i)
                    return [m.cpu_mj / 1000.0] + ([(m.dram_mj or 0.0) / 1000.0] if dram else [])

                self.domains += names
                self._readers.append(read)
                self._gpu_power += [None] * len(names)
            log(f"[energy] zeus CPUs (RAPL): {len(cpus)}")
        except Exception as e:
            log(f"[energy] zeus CPUs (RAPL): unavailable ({type(e).__name__})")
            self._add_windows_emi(log)

        try:
            soc = get_soc()
            probe = soc.get_total_energy_consumption()
            parts = [p for p in APPLE_PARTS if getattr(probe, f"{p}_mj", None) is not None]
            self.domains += [f"apple_{p}" for p in parts]
            self._readers.append(lambda: [getattr(soc.get_total_energy_consumption(), f"{p}_mj") / 1000.0
                                          for p in parts])
            self._gpu_power += [None] * len(parts)
            log(f"[energy] zeus Apple SoC: {parts}")
        except Exception as e:
            log(f"[energy] zeus Apple SoC: unavailable ({type(e).__name__})")

        if not self.domains:
            raise RuntimeError("Zeus found no energy measurement device on this machine")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_counters: Optional[List[float]] = None
        self._last_t: Optional[float] = None
        self.t: List[float] = []
        self.p: List[List[float]] = []  # watts per domain

    def _add_windows_emi(self, log):
        import win_emi

        instances = win_emi.package_instances()
        if not instances:
            log("[energy] Windows Energy Meter: unavailable — no CPU energy on this machine")
            return
        emi = win_emi.EmiEnergyCounter(instances)
        self.domains.append("cpu_pkg")
        self._readers.append(lambda: [emi.read_j()])
        self._gpu_power.append(None)
        log(f"[energy] Windows Energy Meter (RAPL): {instances}")

    # ------------------------------------------------------------------ reading

    def _read_counters(self) -> List[float]:
        out: List[float] = []
        for r in self._readers:
            out.extend(r())
        return out

    def _sample(self):
        """One trace tick: GPU instantaneous power; counter domains from counter deltas."""
        need_counters = any(fn is None for fn in self._gpu_power)
        counters = self._read_counters_non_gpu() if need_counters else None
        now = time.perf_counter()
        vals = []
        for k, fn in enumerate(self._gpu_power):
            if fn is not None:
                try:
                    vals.append(fn())
                except Exception:
                    vals.append(float("nan"))
            elif self._last_counters is not None and self._last_t is not None and now > self._last_t:
                vals.append((counters[k] - self._last_counters[k]) / (now - self._last_t))
            else:
                vals.append(float("nan"))
        with self._lock:
            if counters is not None:
                self._last_counters, self._last_t = counters, now
            self.t.append(now)
            self.p.append(vals)

    def _read_counters_non_gpu(self) -> List[float]:
        """Counters for non-GPU domains only (GPU slots = NaN) — never polls the NVML counter."""
        out: List[float] = []
        k = 0
        for r in self._readers:
            n = self._reader_width(r)
            if all(self._gpu_power[k + j] is not None for j in range(n)):
                out.extend([float("nan")] * n)
            else:
                out.extend(r())
            k += n
        return out

    def _reader_width(self, r) -> int:
        if not hasattr(self, "_widths"):
            self._widths = {id(rr): len(rr()) for rr in self._readers}
        return self._widths[id(r)]

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.period)

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None

    def clear_trace(self):
        with self._lock:
            self.t, self.p = [], []

    # ------------------------------------------------------------------ counters

    def mark(self) -> Mark:
        c = self._read_counters()
        t = time.perf_counter()
        self._sample()
        return Mark(t, c)

    def counter_energy(self, a: Mark, b: Mark) -> Dict[str, float]:
        return {d: y - x for d, x, y in zip(self.domains, a.counters, b.counters)}

    # --------------------------------------------------------------- trace queries

    def trace_energy(self, t0: float, t1: float) -> Dict[str, float]:
        with self._lock:
            ts, ps = list(self.t), list(self.p)
        out = {}
        for k, d in enumerate(self.domains):
            pts = [(t, p[k]) for t, p in zip(ts, ps) if p[k] == p[k]]
            out[d] = _integrate(pts, t0, t1) if len(pts) >= 2 else float("nan")
        return out

    def peak_power(self, t0: float, t1: float) -> Dict[str, float]:
        with self._lock:
            i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
            window = self.p[i:j]
        return {d: max((p[k] for p in window if p[k] == p[k]), default=float("nan"))
                for k, d in enumerate(self.domains)}

    def rows(self, t0: float = float("-inf"), t1: float = float("inf")):
        with self._lock:
            return [{"t": t, **{f"{d}_w": v for d, v in zip(self.domains, p)}}
                    for t, p in zip(self.t, self.p) if t0 <= t <= t1]


def _integrate(pts, t0, t1) -> float:
    ts = [p[0] for p in pts]

    def at(t):
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return pts[0][1]
        if i >= len(pts):
            return pts[-1][1]
        (ta, pa), (tb, pb) = pts[i - 1], pts[i]
        return pa + (pb - pa) * (t - ta) / (tb - ta) if tb > ta else pb

    seq = [(t0, at(t0))] + [p for p in pts if t0 < p[0] < t1] + [(t1, at(t1))]
    return sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(seq, seq[1:]))
