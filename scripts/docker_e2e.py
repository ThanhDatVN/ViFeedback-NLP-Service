"""End-to-end check of the runtime image with the released model mounted (NEXT_PLAN I2).

CI builds the image and smoke-tests it without a model, so readiness there is correctly 503. This
script runs what CI cannot: the image, the real artifact under models/serve, and raw text in.

    python scripts/docker_e2e.py            # build, run, check, remove the container
    python scripts/docker_e2e.py --no-build # reuse vifeedback:e2e

Passes when /readyz is 200, the served file matches the manifest's SHA-256, and the golden cases
come back with their labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE, NAME, PORT = "vifeedback:e2e", "vifeedback-e2e", 18000
BASE = f"http://127.0.0.1:{PORT}"

# Same cases as tests/contract/test_api_with_model.py: negation minimal pairs from the probe.
GOLDEN = [
    ("giảng viên nhiệt tình", "positive"),
    ("giảng viên không nhiệt tình", "negative"),
    ("thầy giảng bài dễ hiểu", "positive"),
    ("thầy giảng bài không dễ hiểu", "negative"),
    ("Giảng viên nhiệt tình", "positive"),  # capitalized input: lowercased by the service
    ("THẦY GIẢNG BÀI KHÔNG DỄ HIỂU", "negative"),
    ("thay giang bai rat de hieu", "positive"),  # unaccented: restored first (ADR-031)
]


def http(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    manifest = json.loads(
        (ROOT / "models/serve/sentiment/manifest.json").read_text(encoding="utf-8")
    )
    if not args.no_build:
        print("building", IMAGE)
        subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(ROOT)], check=True)
    docker("rm", "-f", NAME, check=False)
    docker(
        "run",
        "-d",
        "--name",
        NAME,
        "-p",
        f"{PORT}:8000",
        "-v",
        f"{ROOT / 'models'}:/app/models:ro",
        IMAGE,
    )
    failures: list[str] = []
    try:
        t0, status, ready = time.time(), 0, {}
        while time.time() - t0 < args.timeout:
            try:
                status, ready = http("GET", "/readyz")
                if status == 200:
                    break
            except (urllib.error.URLError, ConnectionError, json.JSONDecodeError):
                pass
            time.sleep(2)
        print(f"/readyz {status} after {time.time() - t0:.0f}s: {ready}")
        if status != 200:
            failures.append(f"/readyz {status}: {ready.get('detail')}")
            print(docker("logs", NAME, check=False).stdout[-2000:])
            return 1

        _, version = http("GET", "/version")
        print("/version", version)
        if manifest["sha256"][:12] not in str(version.get("model_version")):
            failures.append(
                f"/version does not name the served model: {version.get('model_version')}"
            )

        # The file inside the container must be the one the manifest (and the release gate) describe.
        inside = docker(
            "exec",
            NAME,
            "python",
            "-c",
            "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",
            f"/app/models/serve/sentiment/{manifest['model_file']}",
        ).stdout.strip()
        if inside != manifest["sha256"]:
            failures.append(
                f"served file sha256 {inside[:12]} != manifest {manifest['sha256'][:12]}"
            )

        status, out = http("POST", "/v1/classify", {"texts": [t for t, _ in GOLDEN]})
        if status != 200:
            failures.append(f"/v1/classify {status}: {out}")
        else:
            for (text, want), p in zip(GOLDEN, out["predictions"], strict=True):
                ok = p["label"] == want
                print(
                    f"  {'ok ' if ok else 'BAD'} {p['label']:9s} {max(p['probabilities'].values()):.3f}  {text}"
                )
                if not ok:
                    failures.append(f"{text!r}: {p['label']} != {want}")
    finally:
        docker("rm", "-f", NAME, check=False)

    print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
