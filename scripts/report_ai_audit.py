"""Validate an AI-labelled neutral audit and save a text-free exploratory report.

The declared human audit remains separate. This command never writes audit_report.json or changes
the original annotation sheet. Source sentences stay in git-ignored local files.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from vifeedback import paths
from vifeedback.evaluation import audit
from vifeedback.evaluation.report import yaml_safe

LOCAL = paths.RESULTS / "studies" / "study_a" / "local"
SOURCE = LOCAL / "audit_scope_pass1.csv"
AI_SHEET = LOCAL / "audit_scope_ai_provisional.csv"
LABELS = paths.RESULTS / "studies" / "study_a" / "ai_audit_labels.csv"
OUT = paths.RESULTS / "studies" / "study_a" / "ai_audit_report.json"
ANNOTATION_COLUMNS = {"annotator_label", "neutral_subtype", "notes", "gold_assessment"}
LABEL_COLUMNS = ["split", "example_index", "annotator_label", "neutral_subtype", "gold_assessment"]


def main() -> None:
    original = pd.read_csv(SOURCE, encoding="utf-8-sig", keep_default_na=False)
    labels = pd.read_csv(LABELS, encoding="utf-8-sig", keep_default_na=False)
    if len(original) != 58 or len(labels) != 58 or list(labels.columns) != LABEL_COLUMNS:
        raise ValueError("expected the same 58-row neutral audit and text-free label columns")
    keys = ["split", "example_index"]
    if original[keys].duplicated().any() or labels[keys].duplicated().any():
        raise ValueError("duplicate sampled rows")
    if set(map(tuple, original[keys].to_numpy())) != set(map(tuple, labels[keys].to_numpy())):
        raise ValueError("AI label keys differ from the sampled rows")
    if (labels[LABEL_COLUMNS[2:]] == "").any().any():
        raise ValueError("AI labels must be complete")

    if AI_SHEET.exists():
        labelled = pd.read_csv(AI_SHEET, encoding="utf-8-sig", keep_default_na=False)
    else:
        labelled = original.merge(labels, on=keys, validate="one_to_one", suffixes=("", "_ai"))
        for column in LABEL_COLUMNS[2:]:
            labelled[column] = labelled.pop(f"{column}_ai")
        labelled["notes"] = ""
        labelled = labelled[original.columns]
        labelled.to_csv(AI_SHEET, index=False, encoding="utf-8-sig")

    if len(labelled) != 58 or list(original.columns) != list(labelled.columns):
        raise ValueError("expected the same 58-row neutral-audit sheet and columns")
    fixed = [c for c in original.columns if c not in ANNOTATION_COLUMNS]
    if not original[fixed].equals(labelled[fixed]):
        raise ValueError("AI sheet changes sampled rows, text, gold or model metadata")
    if not labelled[LABEL_COLUMNS].equals(
        labels.set_index(keys)
        .loc[pd.MultiIndex.from_frame(labelled[keys])]
        .reset_index()[LABEL_COLUMNS]
    ):
        raise ValueError("local AI sheet differs from the text-free recorded labels")

    result = audit.report(labelled, scope_only=True)
    out = {
        "annotation_provenance": "AI assistant, single provisional pass",
        "independent_human_review": False,
        "decision_status": "exploratory; does not select the preregistered human-audit D2 branch",
        "label_file": str(LABELS.relative_to(paths.ROOT)).replace("\\", "/"),
        "label_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "source_sheet": str(AI_SHEET.relative_to(paths.ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(AI_SHEET.read_bytes()).hexdigest(),
        **result,
    }
    OUT.write_text(
        json.dumps(yaml_safe(out), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    decision = out["decision"]
    print(
        f"{out['rows']} AI-labelled rows; incorrect {decision['share_incorrect']:.1%}, "
        f"ambiguous {decision['share_ambiguous']:.1%}; exploratory branch "
        f"{decision['branch']} -> {OUT}"
    )


if __name__ == "__main__":
    main()
