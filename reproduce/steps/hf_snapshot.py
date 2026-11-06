"""Baixa do Hugging Face, nas revisões fixadas em pins.toml, os modelos e arquivos de uma etapa.

Roda no ambiente `main` (precisa de huggingface_hub). Uso:
  python -m reproduce.steps.hf_snapshot --models Qwen/Qwen2.5-1.5B-Instruct --privacylens DEST.json --gguf
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from reproduce import pins


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--privacylens", help="destino do main_data.json")
    ap.add_argument("--gguf", action="store_true", help="baixa o GGUF do juiz Qwen3-8B")
    a = ap.parse_args()
    p = pins.load()
    for m in a.models:
        rev = p["models"][m]
        print(f"modelo {m} @ {rev}", flush=True)
        snapshot_download(m, revision=rev)
    if a.privacylens:
        ds = "SALT-NLP/PrivacyLens"
        src = hf_hub_download(ds, "main_data.json", repo_type="dataset", revision=p["datasets"][ds])
        if sha256(src) != p["sha256"][f"{ds}/main_data.json"]:
            sys.exit("main_data.json do PrivacyLens com SHA-256 diferente do fixado")
        dest = Path(a.privacylens)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"PrivacyLens -> {dest}")
    if a.gguf:
        repo, name = "Qwen/Qwen3-8B-GGUF", p["tools"]["judge_gguf"]
        path = hf_hub_download(repo, name, revision=p["models"][repo])
        if sha256(path) != p["sha256"][name]:
            sys.exit(f"{name} com SHA-256 diferente do fixado")
        print(f"GGUF -> {path}")


if __name__ == "__main__":
    main()
