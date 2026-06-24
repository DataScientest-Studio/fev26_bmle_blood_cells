"""Échantillonnage CPU/RAM/GPU pendant un entraînement.

Usage :
    monitor = ResourceMonitor()
    monitor.start()
    ... entraînement ...
    summary = monitor.stop()   # dict prêt pour Supabase (training_runs)
"""

import threading
import time

import psutil

try:
    import torch
except ImportError:
    torch = None


class ResourceMonitor:
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._t_start = None
        self._t_end = None

    def _has_cuda(self) -> bool:
        return torch is not None and torch.cuda.is_available()

    def _sample_loop(self):
        proc = psutil.Process()
        while not self._stop_event.is_set():
            sample = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_mb": proc.memory_info().rss / 1e6,
            }
            if self._has_cuda():
                try:
                    sample["gpu_util_percent"] = torch.cuda.utilization()
                except Exception:
                    sample["gpu_util_percent"] = None
                sample["gpu_mem_used_mb"] = torch.cuda.memory_allocated() / 1e6
            self._samples.append(sample)
            self._stop_event.wait(self.interval)

    def start(self):
        self._t_start = time.time()
        psutil.cpu_percent(interval=None)  # amorce le compteur (1er appel toujours 0)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
        self._t_end = time.time()
        return self.summary()

    def summary(self) -> dict:
        duration = (self._t_end or time.time()) - (self._t_start or time.time())

        def avg(key):
            vals = [s[key] for s in self._samples if s.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        result = {
            "duration_seconds": duration,
            "cpu_percent_avg": avg("cpu_percent"),
            "ram_used_mb_avg": avg("ram_used_mb"),
            "gpu_name": None,
            "gpu_util_percent_avg": None,
            "gpu_mem_used_mb_avg": None,
        }
        if self._has_cuda():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_util_percent_avg"] = avg("gpu_util_percent")
            result["gpu_mem_used_mb_avg"] = avg("gpu_mem_used_mb")
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
