"""`vifeedback results`: merge another machine's results, resolve provenance."""

from __future__ import annotations

import json

import typer

from vifeedback import paths
from vifeedback.cli._apps import results_app


@results_app.command("merge")
def results_merge(
    source: str = typer.Argument(..., help="an extracted results folder, e.g. kaggle_results/"),
    dry_run: bool = typer.Option(False, help="report what would be merged, change nothing"),
) -> None:
    """Merge another machine's results into results/: append-only, never overwriting.

    Replaces a manual step that once cost a morning of runs: the Kaggle output was extracted over
    the repository and results/, holding uncommitted local runs, was lost. This command refuses to
    run without a local results/, migrates the registry schema first, appends only run ids not yet
    present, and copies only run directories that do not exist locally.
    """
    import csv
    import shutil
    from pathlib import Path

    import pandas as pd

    from vifeedback.evaluation import report as R

    src = Path(source)
    if not paths.REGISTRY.exists():
        raise typer.BadParameter(
            f"{paths.REGISTRY} is missing. Restore results/ first (git checkout -- results/) instead of "
            "replacing it with another machine's copy; that copy lacks this machine's runs."
        )
    if not (src / "registry.csv").exists():
        raise typer.BadParameter(
            f"{src} has no registry.csv; point at the extracted results folder"
        )

    local = pd.read_csv(paths.REGISTRY)
    other = pd.read_csv(src / "registry.csv")
    new = other[~other.run_id.isin(local.run_id)]
    run_dirs = [
        d
        for d in sorted((src / "runs").glob("*"))
        if d.is_dir() and not (paths.RUNS / d.name).exists()
    ]
    llm_runs = [
        r
        for r in sorted((src / "studies" / "llm_reference").glob("*/*"))
        if r.is_dir()
        and not (paths.RESULTS / "studies" / "llm_reference" / r.parent.name / r.name).exists()
    ]
    typer.echo(f"  registry: {len(new)} new rows ({len(other) - len(new)} already present)")
    typer.echo(f"  run directories: {len(run_dirs)} new")
    typer.echo(f"  llm_reference runs: {len(llm_runs)} new")
    if dry_run:
        for rid in new.run_id:
            typer.echo(f"    + {rid}")
        for r in llm_runs:
            typer.echo(f"    + llm_reference/{r.parent.name}/{r.name}")
        return

    R._migrate_registry_schema()
    with open(paths.REGISTRY, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=R.REGISTRY_FIELDS)
        for _, row in new.iterrows():
            w.writerow(
                {k: ("" if k not in row or pd.isna(row[k]) else row[k]) for k in R.REGISTRY_FIELDS}
            )
    for d in run_dirs:
        shutil.copytree(d, paths.RUNS / d.name)
    manifests = sorted((src / "studies" / "export").glob("*_manifest.json"))
    if manifests:
        dst = paths.RESULTS / "studies" / "export"
        dst.mkdir(parents=True, exist_ok=True)
        for m in manifests:
            if not (dst / m.name).exists():
                shutil.copy(m, dst / m.name)
    # H7 runs (Kaggle Qwen3-4B): one folder per model and configuration; never overwrite one.
    for run in llm_runs:
        shutil.copytree(
            run, paths.RESULTS / "studies" / "llm_reference" / run.parent.name / run.name
        )
        typer.echo(f"  llm_reference: {run.parent.name}/{run.name}")
    typer.echo(
        f"  merged. Now: vifeedback study tables   (registry has {len(pd.read_csv(paths.REGISTRY))} rows)"
    )


@results_app.command("provenance")
def results_provenance() -> None:
    """Resolve every run's recorded source hash to the commit whose code it executed (R11).

    Kaggle runs carry no git SHA; their `source_sha256` is matched against every commit's source.
    Writes results/provenance.json.
    """
    from collections import defaultdict

    from vifeedback.env import source_hash_by_commit

    by_commit = source_hash_by_commit(paths.ROOT)
    runs: dict[str, list[str]] = defaultdict(list)
    for env_file in sorted((paths.RESULTS / "runs").glob("*/env.json")):
        h = json.loads(env_file.read_text(encoding="utf-8")).get("source_sha256") or "none"
        runs[h].append(env_file.parent.name)
    out: dict[str, dict] = {
        h: {
            "first_commit": by_commit[h][0] if h in by_commit else None,
            "commits_with_identical_source": len(by_commit.get(h, [])),
            "note": None if h in by_commit else "no commit has this source: an uncommitted tree",
            "runs": names,
        }
        for h, names in sorted(runs.items())
    }
    (paths.RESULTS / "provenance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for h, r in out.items():
        where = (r["first_commit"] or "uncommitted")[:10]
        typer.echo(f"  {h[:12]}  {len(r['runs']):3d} runs  -> {where}")
