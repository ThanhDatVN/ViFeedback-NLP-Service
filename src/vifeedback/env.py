"""Environment capture for reproducibility.

Every run writes an `env.json`. A latency number without one is an anecdote
(docs/EVALUATION_PROTOCOL.md § Latency harness), and a macro-F1 without one cannot be reproduced.

CPU instruction-set flags matter here beyond bookkeeping: INT8 quantization speedups on CPU depend
heavily on AVX512-VNNI, and without it ONNX Runtime dynamic quantization can be *slower* than FP32
(hypothesis H3, docs/RESEARCH_NOTES.md § 6).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "datasets",
    "onnxruntime",
    "optimum",
    "fastapi",
)


def _pkg_versions() -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for p in _PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return None

    sha = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {
        "sha": sha,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty) if dirty is not None else None,
    }


def cpu_info() -> dict[str, Any]:
    """CPU identity and SIMD capability, with graceful degradation across three sources."""
    import os

    info: dict[str, Any] = {
        "platform_processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "brand": None,
        "physical_cores": None,
        "flags": {},
        "flags_source": None,
    }

    # Preferred: py-cpuinfo gives the real CPUID flag list.
    try:
        import cpuinfo  # type: ignore[import-not-found]

        ci = cpuinfo.get_cpu_info()
        flags = set(ci.get("flags", []))
        info["brand"] = ci.get("brand_raw")
        info["flags"] = {
            f: (f in flags)
            for f in ("avx", "avx2", "fma", "avx512f", "avx512_vnni", "avx_vnni", "sse4_2")
        }
        info["flags_source"] = "py-cpuinfo"
    except Exception:
        pass

    # Fallback: torch reports the widest ISA it dispatches to.
    if not info["flags"]:
        try:
            import torch

            cap = torch.backends.cpu.get_cpu_capability()
            info["flags"] = {"torch_cpu_capability": cap}
            info["flags_source"] = "torch"
        except Exception:
            info["flags_source"] = "unavailable"

    # Windows: WMI gives brand and physical core count.
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "$p=Get-CimInstance Win32_Processor; "
                    "@{name=$p.Name;cores=$p.NumberOfCores} | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=25,
                check=True,
            ).stdout.strip()
            d = json.loads(out)
            info["brand"] = info["brand"] or str(d.get("name", "")).strip() or None
            info["physical_cores"] = d.get("cores")
        except Exception:
            pass

    return info


def has_vnni(info: dict[str, Any] | None = None) -> bool | None:
    """True / False / None (undetermined). Gates the interpretation of INT8 benchmarks."""
    flags = (info or cpu_info()).get("flags", {})
    if "avx512_vnni" in flags or "avx_vnni" in flags:
        return bool(flags.get("avx512_vnni") or flags.get("avx_vnni"))
    cap = flags.get("torch_cpu_capability")
    if isinstance(cap, str):
        return "VNNI" in cap.upper() or "AVX512" in cap.upper()
    return None


def gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda": torch.version.cuda,
        }
    except Exception:
        return {"available": False}


def capture(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cpu = cpu_info()
    env = {
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": cpu,
        "has_vnni": has_vnni(cpu),
        "gpu": gpu_info(),
        "packages": _pkg_versions(),
        "git": _git(),
    }
    if extra:
        env["extra"] = extra
    return env


def write(path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    env = capture(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    return env
