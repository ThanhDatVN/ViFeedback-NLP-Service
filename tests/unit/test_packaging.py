"""Guards against the repository shipping something that does not work.

Written after a real incident: `.gitignore` contained the bare pattern `models/`, intended for the
repo-root directory that holds trained checkpoints. Git matches an unanchored pattern at **any**
depth, so it also matched `src/vifeedback/models/` and silently excluded that entire package from
every commit. The published repository imported fine locally — the files were on disk — and was
broken for anyone who cloned it.

Nothing in the test suite could have caught that, because the tests ran against the working tree.
These run against **git's view** of the repository instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _is_git_repo() -> bool:
    try:
        _git("rev-parse", "--git-dir")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _is_git_repo(), reason="not a git repository")


def _source_files() -> list[Path]:
    return [
        p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts and ".egg-info" not in str(p)
    ]


class TestNoSourceFileIsIgnored:
    def test_every_python_source_file_is_tracked(self) -> None:
        """The exact failure that shipped: source files present locally, absent from the clone."""
        tracked = {
            (ROOT / line).resolve()
            for line in _git("ls-files", "src").splitlines()
            if line.endswith(".py")
        }
        missing = sorted(str(p.relative_to(ROOT)) for p in _source_files() if p not in tracked)
        assert not missing, (
            f"source files exist on disk but are not tracked by git: {missing}. "
            "Check .gitignore for an unanchored pattern - a bare `foo/` matches at any depth."
        )

    def test_no_source_file_matches_a_gitignore_rule(self) -> None:
        files = [str(p.relative_to(ROOT)).replace("\\", "/") for p in _source_files()]
        r = subprocess.run(
            ["git", "check-ignore", "--stdin", "-v"],
            cwd=ROOT,
            input="\n".join(files),
            capture_output=True,
            text=True,
        )
        assert not r.stdout.strip(), f"gitignore rules match source files:\n{r.stdout}"

    def test_every_package_directory_has_an_init(self) -> None:
        """A directory without __init__.py is not importable as a package, which is the other way
        a module can vanish between the working tree and an install."""
        missing = [
            str(d.relative_to(ROOT))
            for d in SRC.rglob("*")
            if d.is_dir()
            and "__pycache__" not in d.parts
            and ".egg-info" not in str(d)
            and any(f.suffix == ".py" for f in d.iterdir() if f.is_file())
            and not (d / "__init__.py").exists()
        ]
        assert not missing, f"package directories without __init__.py: {missing}"


class TestGitignoreHygiene:
    @pytest.mark.parametrize("pattern", ["models", "dist", "build", "data", "results", "docs"])
    def test_directory_patterns_that_could_collide_are_anchored(self, pattern: str) -> None:
        """An unanchored directory pattern whose name could also appear inside `src/` is a trap.

        `models/` was exactly that. Anchoring it (`/models/`) confines it to the repo root.
        """
        lines = [
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        bare = [line for line in lines if line in (pattern, f"{pattern}/")]
        assert not bare, (
            f"unanchored pattern {bare} in .gitignore matches '{pattern}' at any depth. "
            f"Use '/{pattern}/' to confine it to the repository root."
        )


class TestImportability:
    def test_every_module_imports(self) -> None:
        """Catches a module that is tracked but broken - a missing dependency or a syntax error in
        a file no test happens to exercise."""
        import importlib

        failures = []
        for path in _source_files():
            if path.name == "__init__.py":
                module = ".".join(path.relative_to(SRC).parts[:-1])
            else:
                module = ".".join(path.relative_to(SRC).with_suffix("").parts)
            if not module:
                continue
            try:
                importlib.import_module(module)
            except Exception as e:
                failures.append(f"{module}: {type(e).__name__}: {e}")
        assert not failures, "modules failed to import:\n" + "\n".join(failures)
