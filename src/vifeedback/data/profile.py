"""Tokenizer-dependent profiling and the Phase 0 report writer.

Separated from `integrity.py` because these checks need a tokenizer download, while the integrity
suite must stay runnable offline and inside CI.

The subword-length profile is not bookkeeping: it sets `max_length`, and `max_length` is the single
largest lever on FP32 latency available before any optimization work (docs/ROADMAP.md Phase 6, L0-L1).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from vifeedback import env, paths
from vifeedback.constants import MODEL_IDS, SPLITS, TASKS, label_names
from vifeedback.data import integrity as I
from vifeedback.data.loader import load_all

CANDIDATE_MAX_LENGTHS = (32, 64, 96, 128, 256)
_PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.9, 100.0)


def token_length_profile(
    dfs: dict[str, pd.DataFrame], model_key: str = "phobert-base"
) -> dict[str, Any]:
    """Subword-length percentiles and truncation cost at each candidate `max_length`."""
    from transformers import AutoTokenizer

    model_id = MODEL_IDS[model_key]
    tok = AutoTokenizer.from_pretrained(model_id)

    out: dict[str, Any] = {"model": model_id, "vocab_size": int(tok.vocab_size), "splits": {}}
    for split in SPLITS:
        lengths = np.array([len(tok.encode(s)) for s in dfs[split]["sentence"]], dtype=np.int32)
        out["splits"][split] = {
            "mean": round(float(lengths.mean()), 2),
            "percentiles": {f"p{p:g}": int(np.percentile(lengths, p)) for p in _PERCENTILES},
            "truncation": {
                str(ml): {
                    "n": int((lengths > ml).sum()),
                    "share": round(float((lengths > ml).mean()), 5),
                }
                for ml in CANDIDATE_MAX_LENGTHS
            },
        }
    return out


def recommend_max_length(profile: dict[str, Any], tolerance: float = 0.001) -> dict[str, Any]:
    """Smallest candidate `max_length` truncating at most `tolerance` of every split.

    Rationale recorded with the number so the choice is auditable rather than folkloric.
    """
    for ml in CANDIDATE_MAX_LENGTHS:
        shares = [profile["splits"][s]["truncation"][str(ml)]["share"] for s in SPLITS]
        if max(shares) <= tolerance:
            return {
                "max_length": ml,
                "tolerance": tolerance,
                "worst_split_truncation_share": max(shares),
                "vs_model_default_256": round(256 / ml, 2),
                "rationale": (
                    f"Smallest candidate truncating <= {tolerance:.1%} of sentences in every "
                    f"split; {round(256 / ml, 1)}x shorter than the model default of 256."
                ),
            }
    return {
        "max_length": CANDIDATE_MAX_LENGTHS[-1],
        "tolerance": tolerance,
        "rationale": "No candidate met the tolerance; falling back to the model maximum.",
    }


def teencode_presence(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """How much informal orthography the corpus actually contains.

    Sizes the TEENCODE error category *before* modeling. If the benchmark turns out to be largely
    clean, the teencode work is a deployment-robustness argument rather than a benchmark-accuracy
    one — a distinction that changes how the result must be described.
    """
    probes = {
        "k": "không",
        "ko": "không",
        "hok": "không",
        "kg": "không",
        "kh": "không",
        "j": "gì",
        "z": "vậy",
        "dz": "vậy",
        "vs": "với",
        "dc": "được",
        "đc": "được",
        "bt": "bình thường",
        "sv": "sinh viên",
        "gv": "giảng viên",
        "mn": "mọi người",
        "ntn": "như thế nào",
        "nt": "nhắn tin",
        "ok": "ổn",
        "wa": "quá",
        "wá": "quá",
    }
    full = pd.concat([dfs[s] for s in SPLITS], ignore_index=True)
    tokens = full["sentence"].str.split()

    counts: dict[str, int] = {}
    for probe in probes:
        counts[probe] = int(tokens.map(lambda t, p=probe: p in t).sum())

    any_probe = tokens.map(lambda t: bool(set(t) & set(probes))).sum()
    return {
        "probe_dictionary_size": len(probes),
        "sentences_with_any_probe": int(any_probe),
        "share_of_corpus": round(float(any_probe / len(full)), 4),
        "per_probe": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


def surface_properties(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Corpus-level orthographic facts that determine which ablation conditions are meaningful.

    UIT-VSFC ships pre-lowercased and pre-tokenized, which makes some textbook preprocessing steps
    no-ops here. Measuring that is cheaper than running the ablation and finding a flat line.
    """
    full = pd.concat([dfs[s] for s in SPLITS], ignore_index=True)
    s = full["sentence"]
    has_upper = (s != s.str.lower()).sum()
    return {
        "n": len(s),
        "sentences_with_uppercase": int(has_upper),
        "uppercase_share": round(float(has_upper / len(s)), 4),
        "sentences_with_underscore": int(s.str.contains("_", regex=False).sum()),
        "sentences_with_space_before_final_period": int(s.str.endswith(" .").sum()),
        "space_before_final_period_share": round(float(s.str.endswith(" .").mean()), 4),
        "sentences_with_digits": int(s.str.contains(r"\d", regex=True).sum()),
    }


def build_report(with_tokenizer: bool = True) -> dict[str, Any]:
    dfs = load_all()
    report: dict[str, Any] = {
        "environment": env.capture(),
        "integrity": I.full_report(dfs),
        "surface": surface_properties(dfs),
        "teencode": teencode_presence(dfs),
    }
    if with_tokenizer:
        prof = token_length_profile(dfs)
        report["token_lengths"] = prof
        report["max_length_decision"] = recommend_max_length(prof)
    return report


def write_report(report: dict[str, Any]) -> None:
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out = paths.RESULTS / "data_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Figures ------------------------------------------------------------------------------------


def make_figures(dfs: dict[str, pd.DataFrame] | None = None) -> list[str]:
    """EDA figures for the data card. Regenerated from data, never hand-made."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dfs = dfs or load_all()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. Class distribution per split, both tasks.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, task in zip(axes, TASKS, strict=True):
        names = label_names(task)
        dist = I.class_distribution(dfs, task).set_index("split")
        share = dist.loc[list(SPLITS), [f"{n}_pct" for n in names]]
        share.columns = names
        share.plot(kind="bar", ax=ax, rot=0, width=0.78, edgecolor="white", linewidth=0.6)
        ax.set_title(f"{task} — class share by split (%)")
        ax.set_ylabel("% of split")
        ax.set_xlabel("")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = paths.FIGURES / "dist_classes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    # 2. Sentence-length distribution (syllables), train.
    from vifeedback.preprocess import syllable_count

    fig, ax = plt.subplots(figsize=(7.5, 4))
    syl = dfs["train"]["sentence"].map(syllable_count)
    ax.hist(syl, bins=range(0, 80), edgecolor="white", linewidth=0.4)
    for q, style in ((50, ":"), (95, "--"), (99, "-.")):
        v = float(np.percentile(syl, q))
        ax.axvline(v, linestyle=style, linewidth=1.2, color="crimson")
        ax.text(v + 0.6, ax.get_ylim()[1] * 0.82, f"p{q}={v:.0f}", fontsize=8, color="crimson")
    ax.set_title("UIT-VSFC train — sentence length (syllables)")
    ax.set_xlabel("syllables")
    ax.set_ylabel("sentences")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = paths.FIGURES / "dist_length.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    # 3. Sentiment x topic heatmap — the multi-task evidence.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    j = I.joint_distribution(dfs)
    pct = (100 * j / j.values.sum()).round(1)
    im = ax.imshow(pct.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(j.columns)), j.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(j.index)), j.index, fontsize=9)
    for r in range(len(j.index)):
        for c in range(len(j.columns)):
            ax.text(
                c,
                r,
                f"{j.values[r, c]}\n{pct.values[r, c]}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if pct.values[r, c] < pct.values.max() * 0.6 else "black",
            )
    ax.set_title(f"sentiment x topic (full corpus) — Cramér's V = {I.cramers_v(j):.3f}")
    fig.colorbar(im, ax=ax, label="% of corpus")
    fig.tight_layout()
    p = paths.FIGURES / "dist_joint.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    return written
