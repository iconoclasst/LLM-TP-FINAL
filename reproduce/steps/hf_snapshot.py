"""Baixa do Hugging Face, nas revisões fixadas em pins.toml, os modelos e arquivos de uma etapa.

Os scripts dos três trabalhos carregam a revisão `main`. Por isso, depois de baixar cada repositório
na revisão fixada, gravamos essa revisão em `refs/main` no cache; as etapas seguintes rodam com
HF_HUB_OFFLINE=1 e resolvem `main` para a revisão fixada, sem consultar o Hub.

Roda no ambiente `main` (precisa de huggingface_hub). Uso:
  python -m reproduce.steps.hf_snapshot --models Qwen/Qwen2.5-1.5B-Instruct --privacylens DEST.json --gguf --csqa
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import constants, hf_hub_download, snapshot_download

from reproduce import pins


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin_main(repo: str, repo_type: str, revision: str) -> None:
    folder = Path(constants.HF_HUB_CACHE) / f"{repo_type}s--{repo.replace('/', '--')}" / "refs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main").write_text(revision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--privacylens", help="destino do main_data.json")
    ap.add_argument("--gguf", action="store_true", help="baixa o GGUF do juiz Qwen3-8B")
    ap.add_argument("--csqa", action="store_true", help="baixa o CommonSenseQA (tau/commonsense_qa)")
    a = ap.parse_args()
    p = pins.load()
    for m in a.models:
        rev = p["models"][m]
        print(f"modelo {m} @ {rev}", flush=True)
        snapshot_download(m, revision=rev)
        pin_main(m, "model", rev)
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
        pin_main(repo, "model", p["models"][repo])
        print(f"GGUF -> {path}")
    if a.csqa:
        ds = "tau/commonsense_qa"
        rev = p["hf_datasets"][ds]
        snapshot_download(ds, repo_type="dataset", revision=rev)
        pin_main(ds, "dataset", rev)
        from datasets import load_dataset   # prepara o cache do datasets para o uso offline
        n = len(load_dataset(ds, split="validation", revision=rev))
        print(f"CommonSenseQA @ {rev}: {n} itens de validação")


if __name__ == "__main__":
    main()
