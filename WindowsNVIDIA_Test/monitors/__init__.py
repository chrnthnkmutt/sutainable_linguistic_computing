from .apple_cpu import AppleCPUMonitor
from .apple_gpu import AppleGPUMonitor
from .base import BaseMonitor
from .intel_cpu import IntelCPUMonitor
from .memory import MemoryMonitor
from .nvidia_gpu import NvidiaGPUMonitor

# Order = order shown in the terminal menu.
ALL_MONITORS = [IntelCPUMonitor, NvidiaGPUMonitor, AppleCPUMonitor, AppleGPUMonitor]

__all__ = ["ALL_MONITORS", "BaseMonitor", "MemoryMonitor", "IntelCPUMonitor",
           "NvidiaGPUMonitor", "AppleCPUMonitor", "AppleGPUMonitor"]
