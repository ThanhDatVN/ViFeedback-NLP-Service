"""The root command and its groups. Command modules register on these; `__init__` imports them."""

from __future__ import annotations

import typer

# Tracebacks never print local variables: a failing API call holds the key in its locals (.env).
app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="ViFeedback — Vietnamese feedback classification",
)


@app.callback()
def _startup() -> None:
    """ViFeedback — Vietnamese feedback classification."""
    from vifeedback.env import load_dotenv

    load_dotenv()  # keys from the git-ignored .env (OPENAI_API_KEY, HF_TOKEN); terminal values win


data_app = typer.Typer(
    pretty_exceptions_show_locals=False, help="Phase 0: acquisition, integrity, profiling"
)
baseline_app = typer.Typer(pretty_exceptions_show_locals=False, help="Phase 1: classical baselines")
train_app = typer.Typer(
    pretty_exceptions_show_locals=False, help="Phase 2+: transformer fine-tuning"
)
serve_app = typer.Typer(
    pretty_exceptions_show_locals=False, help="Phase 6-7: export, benchmark, serve"
)
study_app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Research studies (docs/REVIEW_AND_RESEARCH_PLAN.md § 7)",
)
results_app = typer.Typer(pretty_exceptions_show_locals=False, help="Results housekeeping")

app.add_typer(data_app, name="data")
app.add_typer(baseline_app, name="baseline")
app.add_typer(train_app, name="train")
app.add_typer(serve_app, name="serve")
app.add_typer(study_app, name="study")
app.add_typer(results_app, name="results")
