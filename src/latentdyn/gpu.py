"""GPU occupancy helpers (read-only nvidia-smi queries). Never selects or sets devices itself."""
from __future__ import annotations

import os
import subprocess
from typing import List, Optional


def smi() -> dict:
    """{gpu_index: (memory_used_MiB, utilization_pct)} (from vfield_watch.py)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(l.split(",")[0]): (int(l.split(",")[1]), int(l.split(",")[2])) for l in out.strip().splitlines()}


def idle_gpus(mem_mib: int = 500, util_pct: int = 2, exclude: tuple = (0,)) -> List[int]:
    """Indices with memory.used < mem_mib and utilization <= util_pct, excluding `exclude` (GPU 0 has an ECC fault)."""
    return [g for g, (mem, util) in sorted(smi().items()) if g not in exclude and mem < mem_mib and util <= util_pct]


def pinned_gpu_indices() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip() != ""]


def min_free_mib_on_pinned_gpus(tag: str = "run_sweep") -> Optional[int]:
    """Minimum free MiB across the GPU indices pinned via CUDA_VISIBLE_DEVICES.

    Returns None if CUDA_VISIBLE_DEVICES is unset (nothing pinned -- e.g. a
    CPU-only dry run) or if the query fails for any reason (fail-open: we do
    not want a flaky nvidia-smi call to block a sweep forever).
    """
    indices = pinned_gpu_indices()
    if not indices:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
                "-i", ",".join(indices),
            ],
            text=True, timeout=10,
        )
    except Exception as e:  # noqa: BLE001 - deliberately broad, fail-open guard
        print(f"[{tag}] WARNING: nvidia-smi free-memory query failed ({e}); skipping GPU memory guard.")
        return None

    free_values = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                free_values.append(int(parts[1]))
            except ValueError:
                continue
    return min(free_values) if free_values else None
