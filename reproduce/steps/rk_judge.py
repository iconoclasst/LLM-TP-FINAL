"""Roda o run_judge.py do Rubens/Kayque apontando para o llama-server e o GGUF fixados.

Roda no ambiente `main`, com cwd na pasta de scripts do Rubens/Kayque. LLAMA_SERVER e JUDGE_GGUF
definidos pelo usuário têm prioridade.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from reproduce import pins
from reproduce.steps.llama_cpp import find_server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-dir", required=True)
    a = ap.parse_args()
    env = dict(os.environ)
    if not env.get("LLAMA_SERVER"):
        srv = find_server(Path(a.llama_dir))
        if srv is None:
            sys.exit(f"llama-server não encontrado em {a.llama_dir}")
        env["LLAMA_SERVER"] = str(srv)
        # as bibliotecas compartilhadas do llama.cpp ficam ao lado do executável
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[var] = os.pathsep.join([str(srv.parent), env.get(var, "")]).rstrip(os.pathsep)
    if not env.get("JUDGE_GGUF"):
        from huggingface_hub import hf_hub_download
        p = pins.load()
        repo = "Qwen/Qwen3-8B-GGUF"
        env["JUDGE_GGUF"] = hf_hub_download(repo, p["tools"]["judge_gguf"], revision=p["models"][repo])
    sys.exit(subprocess.call([sys.executable, "run_judge.py"], env=env))


if __name__ == "__main__":
    main()
