"""Hardware and library versions, recorded next to every measured result."""

from __future__ import annotations

import platform
import subprocess
import sys


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            if out:
                return out
        elif sys.platform == "linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def system_info(device: str | None = None) -> dict:
    import cv2
    import numpy
    import torch
    import ultralytics

    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    if device is not None:
        info["device_used"] = "cpu" if str(device) == "cpu" else info.get("gpu", f"cuda:{device}")
    return info
