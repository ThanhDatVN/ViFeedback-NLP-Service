"""`python -m vifeedback.cli ...`, the form the Makefile, CI and the Kaggle notebooks use."""

from vifeedback.cli import app

# Importing this module (the packaging test imports every module) must not run the CLI.
if __name__ == "__main__":
    app()
