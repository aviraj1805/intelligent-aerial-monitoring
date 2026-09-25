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

    os_name = f"{platform.system()} {platform.release()}"
    if platform.system() == "Windows":
        # Windows 11 still reports release "10"; builds >= 22000 are Windows 11.
        build = int(platform.version().split(".")[-1])
        os_name = f"Windows {'11' if build >= 22000 else platform.release()} (build {build})"
    info = {
        "os": os_name,
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
        battery = psutil.sensors_battery()
        if battery is not None:  # laptops throttle on battery; record it
            info["power"] = "AC adapter" if battery.power_plugged else "battery"
    except ImportError:
        pass
    try:  # Ultralytics hides the GPU (CUDA_VISIBLE_DEVICES) when device="cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except (AssertionError, RuntimeError):
        pass
    if device is not None:
        info["device_used"] = "cpu" if str(device) == "cpu" else info.get("gpu", f"cuda:{device}")
        if str(device) == "cpu":
            info.pop("gpu", None)
            info.pop("gpu_mem_gb", None)
    return info
