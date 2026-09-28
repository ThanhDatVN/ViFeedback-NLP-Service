"""serve reproduce: the download step verifies every file (NEXT_PLAN v5 F2). No network: a fake Hub."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from vifeedback.inference import reproduce as RP


def _release(src, tamper: str | None = None):
    files = {
        "model.opt.onnx": b"graph",
        "restorer.json": b"{}",
        "vocab.txt": b"a 1\n",
        "bpe.codes": b"#version\n",
        "added_tokens.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for name, data in files.items():
        (src / name).write_bytes(data)
    manifest = {
        "model_file": "model.opt.onnx",
        "sha256": hashlib.sha256(files["model.opt.onnx"]).hexdigest(),
        "restorer": {"file": "restorer.json"},
    }
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    names = [*files, "manifest.json"]
    sums = "".join(f"{hashlib.sha256((src / n).read_bytes()).hexdigest()}  {n}\n" for n in names)
    (src / "SHA256SUMS").write_text(sums, encoding="utf-8")
    if tamper:
        (src / tamper).write_bytes(b"changed after the sums were written")


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    import huggingface_hub

    src, models = tmp_path / "hub", tmp_path / "models"
    src.mkdir()

    def download(repo_id, name, revision=None, local_dir=None):
        dst = local_dir / name
        shutil.copy(src / name, dst)
        return str(dst)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(RP.paths, "MODELS", models)
    return src


def test_a_verified_release_downloads(fake_hub):
    _release(fake_hub)
    d, verified = RP.download("owner/name")
    assert (d / "model.opt.onnx").exists() and verified["files_checked"] == 7


def test_a_tampered_file_is_refused(fake_hub):
    _release(fake_hub, tamper="vocab.txt")
    with pytest.raises(ValueError, match=r"vocab.txt does not match SHA256SUMS"):
        RP.download("owner/name")


def test_a_model_that_is_not_the_manifests_is_refused(fake_hub):
    _release(fake_hub)
    m = json.loads((fake_hub / "manifest.json").read_text(encoding="utf-8"))
    m["sha256"] = "0" * 64
    (fake_hub / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError):  # the manifest no longer matches SHA256SUMS either
        RP.download("owner/name")
