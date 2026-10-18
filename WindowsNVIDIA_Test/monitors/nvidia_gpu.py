"""NVIDIA GPU via NVML (nvidia-ml-py): utilisation, power, VRAM and the total-energy counter."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import BaseMonitor


def _decode(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


class NvidiaGPUMonitor(BaseMonitor):
    key = "nvidia_gpu"
    label = "NVIDIA GPU"
    kind = "gpu"
    gpu_index = 0
    _name = "NVIDIA GPU"

    @classmethod
    def detect(cls) -> Tuple[bool, str]:
        try:
            import pynvml
        except ImportError:
            return False, "nvidia-ml-py (pynvml) is not installed"
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                return False, "no NVIDIA GPU found"
            cls._name = _decode(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(cls.gpu_index)))
            return True, cls._name
        except Exception as exc:
            return False, f"NVML unavailable: {exc}"

    @classmethod
    def hardware_name(cls) -> str:
        return cls._name

    def _begin(self) -> None:
        import pynvml

        self._nv = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._vram_start = pynvml.nvmlDeviceGetMemoryInfo(self._h).used / 1048576
        try:
            self._e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)  # millijoules
            self._has_counter = True
        except pynvml.NVMLError:
            self._has_counter = False
            self._note("NVML energy counter unsupported (pre-Volta GPU?): energy integrated from power samples")

    def _sample(self) -> Optional[Dict[str, Any]]:
        nv, h = self._nv, self._h
        power = None
        try:
            power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        except nv.NVMLError:
            self._note("GPU power reading unsupported")
        return {
            "usage": float(nv.nvmlDeviceGetUtilizationRates(h).gpu),
            "power": power,
            "vram_mb": nv.nvmlDeviceGetMemoryInfo(h).used / 1048576,
        }

    def _end(self) -> None:
        if self._has_counter:
            self._energy_j = (self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h) - self._e0) / 1000.0
        else:
            pts = [(s["t"], s["power"]) for s in self.samples if s.get("power") is not None]
            if pts:
                pts = [(0.0, pts[0][1])] + pts
                self._energy_j = sum((t1 - t0) * (p1 + p0) / 2 for (t0, p0), (t1, p1) in zip(pts, pts[1:]))
        vram = [s["vram_mb"] for s in self.samples if "vram_mb" in s]
        self.extra.update(
            vram_start_mb=self._vram_start,
            vram_end_mb=vram[-1] if vram else None,
            vram_peak_mb=max(vram + [self._vram_start]),
        )
