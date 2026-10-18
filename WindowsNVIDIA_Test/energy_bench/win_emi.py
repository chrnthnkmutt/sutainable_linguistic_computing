"""Windows Energy Meter Interface (EMI): CPU package energy without admin rights or extra drivers.

Windows 10/11 exposes the CPU's RAPL counters (Intel and AMD) through EMI, readable as the
"Energy Meter" performance-counter set: instances like RAPL_Package0_PKG and
RAPL_Package0_Core<N>_CORE, counter "Energy" = cumulative energy in picowatt-hours.

Checked on a Ryzen 5 7500F: idle 41 W, all-core load 89 W (= its 88 W PPT limit); the counter
updates every ~6 ms. Read with PDH through ctypes (no pywin32 dependency).
"""
from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from typing import List, Optional

PWH_TO_J = 3.6e-9  # 1 pWh = 3.6e-9 J


class _RawCounter(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("TimeStamp", ctypes.c_ulonglong),
                ("FirstValue", ctypes.c_longlong), ("SecondValue", ctypes.c_longlong),
                ("MultiCount", ctypes.c_ulong)]


@lru_cache(maxsize=1)
def package_instances() -> List[str]:
    """EMI package meters on this machine, e.g. ['RAPL_Package0_PKG'] ([] if EMI is unavailable)."""
    if sys.platform != "win32":
        return []
    pdh = ctypes.WinDLL("pdh")
    counters, instances = ctypes.c_ulong(0), ctypes.c_ulong(0)
    pdh.PdhEnumObjectItemsW(None, None, "Energy Meter", None, ctypes.byref(counters), None,
                            ctypes.byref(instances), 400, 0)  # PERF_DETAIL_WIZARD
    if instances.value == 0:
        return []
    cbuf = ctypes.create_unicode_buffer(counters.value)
    ibuf = ctypes.create_unicode_buffer(instances.value)
    if pdh.PdhEnumObjectItemsW(None, None, "Energy Meter", cbuf, ctypes.byref(counters), ibuf,
                               ctypes.byref(instances), 400, 0) != 0:
        return []
    names = [n for n in ctypes.wstring_at(ctypes.addressof(ibuf), instances.value).split("\0") if n]
    return sorted(n for n in names if n.upper().endswith("_PKG"))


class EmiEnergyCounter:
    """Cumulative joules over the given EMI meter instances."""

    def __init__(self, instances: List[str]):
        self._pdh = ctypes.WinDLL("pdh")
        self._query = ctypes.c_void_p()
        self._counters: List[ctypes.c_void_p] = []
        if self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)) != 0:
            raise OSError("PdhOpenQuery failed")
        for inst in instances:
            c = ctypes.c_void_p()
            r = self._pdh.PdhAddEnglishCounterW(self._query, f"\\Energy Meter({inst})\\Energy", None, ctypes.byref(c))
            if r != 0:
                raise OSError(f"Energy Meter({inst}) not readable (PDH 0x{r & 0xFFFFFFFF:08x})")
            self._counters.append(c)
        self._start: Optional[int] = None
        self._start = self._raw()

    def _raw(self) -> int:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            raise OSError("PdhCollectQueryData failed")
        total = 0
        for c in self._counters:
            raw, typ = _RawCounter(), ctypes.c_ulong()
            if self._pdh.PdhGetRawCounterValue(c, ctypes.byref(typ), ctypes.byref(raw)) != 0:
                raise OSError("PdhGetRawCounterValue failed")
            total += raw.FirstValue
        return total

    def read_j(self) -> float:
        """Joules since this counter was created."""
        return (self._raw() - self._start) * PWH_TO_J

    def close(self) -> None:
        self._pdh.PdhCloseQuery(self._query)
