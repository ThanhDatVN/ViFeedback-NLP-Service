"""ViFeedback — Vietnamese feedback sentiment & topic classification.

See docs/ROADMAP.md for the phase plan and docs/EVALUATION_PROTOCOL.md for the measurement contract.

`__version__` is load-bearing beyond bookkeeping: if `import vifeedback` ever yields an object
without it, the import resolved to a PEP 420 **namespace package** — a directory named `vifeedback`
on `sys.path` shadowing this one — and every submodule import after that will fail confusingly.
The check cannot live in this file (a namespace package never executes it), so entry points that
run outside the package do it: see `check_import()` below, called by the notebooks.
"""

from __future__ import annotations

__version__ = "0.1.0"


def check_import() -> str:
    """Verify this import is the real package. Returns the version.

    Call this from a notebook or script **after** installing. It is deliberately trivial, because
    the failure it catches is not: on Kaggle the repository was extracted to a directory named
    `vifeedback`, PEP 420 turned that directory into the package, `vifeedback.data` resolved to the
    *dataset folder*, and `vifeedback.evaluation` raised ModuleNotFoundError forty minutes into a
    GPU run. The traceback named neither the cause nor the fix.
    """
    import sys

    mod = sys.modules[__name__]
    if getattr(mod, "__file__", None) is None:
        raise ImportError(
            "`vifeedback` resolved to a NAMESPACE package, not the installed library. A directory "
            "named 'vifeedback' on sys.path is shadowing it — most likely this repository was "
            "extracted or cloned into a folder of that name. Rename it (e.g. to 'repo') and re-run. "
            f"sys.path[:3] = {sys.path[:3]}"
        )
    return __version__
