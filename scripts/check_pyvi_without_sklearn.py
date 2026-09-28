"""NEXT_PLAN v5 F1: pyvi segments identically without scikit-learn (and so without SciPy).

pyvi's CRF is a pickled `sklearn_crfsuite.CRF`, whose base class comes from scikit-learn only when
it is installed (`sklearn_crfsuite.compat` falls back to a plain class). If segmentation is
byte-identical with scikit-learn blocked, the runtime image can drop scikit-learn and SciPy.

    python scripts/check_pyvi_without_sklearn.py

Segments every UIT-VSFC sentence (all splits) and every NEU-ESC post, once normally and once in a
subprocess where `import sklearn` fails, and writes the comparison (counts only, no text) to
results/studies/engineering/pyvi_without_sklearn.json. Exit code 1 on any difference.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHILD = """
import sys, json
sys.modules["sklearn"] = None  # any `import sklearn...` now raises ImportError
from pyvi import ViTokenizer
import sklearn_crfsuite.compat as c
assert c.BaseEstimator.__module__ == "sklearn_crfsuite.compat", "scikit-learn was imported"
texts = json.load(open(sys.argv[1], encoding="utf-8"))
json.dump([ViTokenizer.tokenize(t) for t in texts], open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
"""


def main() -> int:
    from pyvi import ViTokenizer

    from vifeedback.data.loader import load
    from vifeedback.evaluation import external as X
    from vifeedback.preprocess.normalize import model_text

    texts = [t for s in ("train", "validation", "test") for t in load(s).sentence.tolist()]
    n_uit = len(texts)
    try:
        texts += [model_text(t) for s in ("train", "validation", "test") for t in X.load_neu_esc(s).text]
    except FileNotFoundError:
        print("NEU-ESC is not fetched; checking UIT-VSFC only")
    normal = [ViTokenizer.tokenize(t) for t in texts]

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        subprocess.run([sys.executable, "-c", CHILD, str(src), str(dst)], check=True)
        blocked = json.loads(dst.read_text(encoding="utf-8"))

    differ = sum(a != b for a, b in zip(normal, blocked, strict=True))
    digest = hashlib.sha256("\n".join(normal).encode()).hexdigest()
    out = {
        "question": "does pyvi segment identically without scikit-learn? (NEXT_PLAN v5 F1)",
        "texts": len(texts),
        "uit_vsfc": n_uit,
        "neu_esc": len(texts) - n_uit,
        "different": differ,
        "identical": differ == 0,
        "sha256_of_segmented_output": digest,
    }
    dst_json = ROOT / "results" / "studies" / "engineering" / "pyvi_without_sklearn.json"
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if differ == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
