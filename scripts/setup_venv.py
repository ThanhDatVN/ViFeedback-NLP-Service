"""Create the project's virtual environment in .venv with the pinned package versions.

    python scripts/setup_venv.py            # CUDA torch if an NVIDIA GPU is present, else CPU torch
    python scripts/setup_venv.py --cpu      # force CPU torch
    python scripts/setup_venv.py --recreate # delete .venv first

Installs the package editable with the `all` extra (train, serve, export, dev, bench), constrained by
requirements-lock.txt, the environment every laptop result came from (review R11). `seg` (VnCoreNLP,
underthesea) is left out; add it with `.venv/Scripts/python -m pip install -e ".[seg]"`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
LOCK = ROOT / "requirements-lock.txt"
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu126",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def has_nvidia_gpu() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run(*cmd: str) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="install CPU torch even if a GPU is present")
    ap.add_argument("--recreate", action="store_true", help="delete an existing .venv first")
    args = ap.parse_args()

    if sys.version_info < (3, 11):  # noqa: UP036 - this script may be run by any python
        print(f"Python >= 3.11 is required, this is {sys.version.split()[0]}")
        return 1
    if args.recreate and VENV.exists():
        shutil.rmtree(VENV)
    if not venv_python().exists():
        print(f"creating {VENV}")
        run(sys.executable, "-m", "venv", str(VENV))

    flavour = "cpu" if args.cpu or not has_nvidia_gpu() else "cuda"
    lock = LOCK.read_text(encoding="utf-8")
    # The lock pins the reference laptop's CUDA build; the CPU wheel carries another local label.
    if flavour == "cpu":
        lock = re.sub(r"^torch==([\d.]+)\+\w+$", r"torch==\1", lock, flags=re.M)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(lock)
        constraints = f.name

    py = str(venv_python())
    run(py, "-m", "pip", "install", "--upgrade", "pip")
    run(
        py,
        "-m",
        "pip",
        "install",
        "-e",
        ".[all]",
        "-c",
        constraints,
        "--extra-index-url",
        TORCH_INDEX[flavour],
    )
    Path(constraints).unlink(missing_ok=True)

    check = (
        "import torch, transformers, onnxruntime, fastapi, pyvi, vifeedback;"
        "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),"
        " '| transformers', transformers.__version__, '| onnxruntime', onnxruntime.__version__)"
    )
    run(py, "-c", check)
    act = r".venv\Scripts\Activate.ps1" if sys.platform == "win32" else "source .venv/bin/activate"
    print(f"\nready ({flavour} torch). Activate with: {act}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
