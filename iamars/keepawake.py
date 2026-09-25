"""Keep Windows from idle-sleeping while a long job (training, evaluation) runs.

Uses SetThreadExecutionState, which lasts only while the process runs and does
not change the user's power settings. No-op on other operating systems.
"""

import contextlib
import sys


@contextlib.contextmanager
def keep_awake():
    if sys.platform != "win32":
        yield
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
