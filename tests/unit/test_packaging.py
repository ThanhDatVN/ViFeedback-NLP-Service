"""Guards against the repository shipping something that does not work.

Two real incidents are encoded here.

**1. A `.gitignore` pattern excluded a source package.** The file held the bare pattern `models/`,
intended for the repo-root directory of trained checkpoints. Git matches an unanchored pattern at
*any* depth, so it also matched `src/vifeedback/models/` and silently excluded that package from
every commit. The published repository imported fine locally — the files were on disk — and was
broken for anyone who cloned it. No existing test could catch it, because the tests ran against the
working tree. These run against **git's view** of the repository.

**2. A directory named `vifeedback` shadowed the installed package.** On Kaggle the repo was
extracted to `/kaggle/working/vifeedback`, whose parent was on `sys.path`. PEP 420 made that
directory the package, so `vifeedback.data` resolved to the *dataset folder* and
`vifeedback.evaluation` raised ModuleNotFoundError forty minutes into a GPU run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
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


def _source_files() -> list[Path]:
    return [
        p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts and ".egg-info" not in str(p)
    ]


@pytest.mark.skipif(not _is_git_repo(), reason="not a git repository")
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
        """A directory without __init__.py is not importable as a package - the other way a module
        can vanish between the working tree and an install."""
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
    @pytest.mark.skipif(not _is_git_repo(), reason="not a git repository")
    @pytest.mark.parametrize("pattern", ["models", "dist", "build", "data", "results", "docs"])
    def test_directory_patterns_that_could_collide_are_anchored(self, pattern: str) -> None:
        """An unanchored directory pattern whose name could also appear inside `src/` is a trap.

        `models/` was exactly that. Anchoring it (`/models/`) confines it to the repository root.
        """
        lines = [
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        bare = [line for line in lines if line in (pattern, f"{pattern}/")]
        assert not bare, (
            f"unanchored pattern {bare} matches '{pattern}' at any depth. "
            f"Use '/{pattern}/' to confine it to the repository root."
        )


class TestImportability:
    def test_every_module_imports(self) -> None:
        """Catches a module that is tracked but broken - a missing dependency or a syntax error in
        a file no other test happens to exercise."""
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


class TestNamespaceShadowing:
    """A directory named `vifeedback` on sys.path silently replaces the installed package."""

    def test_the_installed_package_is_not_a_namespace_package(self) -> None:
        import vifeedback

        assert vifeedback.__file__ is not None, (
            "vifeedback resolved to a namespace package - a directory of that name is on sys.path"
        )
        assert vifeedback.check_import() == vifeedback.__version__

    def test_shadowing_reproduces_the_exact_kaggle_failure(self, tmp_path) -> None:
        """Reproduce it, and assert the two symptoms that made it hard to diagnose.

        Writing this test established something worth recording: a guard placed *inside*
        `__init__.py` would be dead code, because a namespace package never executes one. The
        check has to run from outside, which is why `check_import()` exists and why the notebooks
        call it explicitly rather than relying on an import side effect.

        The subprocess runs with `-S` (no `site`), which disables the `.pth` files and meta-path
        finders an editable install relies on. That is what reproduces Kaggle's configuration: a
        PEP 660 editable install registers a finder that runs *before* `sys.path` and is immune to
        this shadowing, and Kaggle's install was not.
        """
        shadow = tmp_path / "vifeedback"
        (shadow / "data").mkdir(parents=True)
        (shadow / "docs").mkdir(parents=True)

        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            import vifeedback
            print("FILE:", vifeedback.__file__)
            import vifeedback.data
            print("DATA:", vifeedback.data.__path__[0])
            try:
                import vifeedback.evaluation
                print("EVAL: imported")
            except ModuleNotFoundError as e:
                print("EVAL:", e)
            """
        )
        r = subprocess.run(
            [sys.executable, "-S", "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(tmp_path),
        )

        assert "FILE: None" in r.stdout, (
            "the shadowing scenario no longer reproduces, so this test guards nothing. "
            f"stdout={r.stdout!r} stderr={r.stderr[-400:]!r}"
        )
        assert str(shadow / "data") in r.stdout, "vifeedback.data resolved to the shadow directory"
        assert "No module named 'vifeedback.evaluation'" in r.stdout, (
            "this is the exact error the Kaggle run hit after 40 minutes of GPU time"
        )

    @pytest.mark.skipif(not _is_git_repo(), reason="not a git repository")
    def test_repository_root_is_not_named_vifeedback(self) -> None:
        """A clone into a folder of that name breaks every notebook in this repository."""
        assert ROOT.name.lower() != "vifeedback", (
            f"the repository is checked out at {ROOT}, whose name shadows the package. Rename it."
        )
