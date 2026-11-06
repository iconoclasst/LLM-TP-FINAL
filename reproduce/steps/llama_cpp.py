"""Baixa e extrai o llama.cpp fixado (build b11505) para a plataforma atual. Só biblioteca padrão + curl."""
import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from reproduce import pins


def platform_key():
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def find_server(root: Path):
    names = ("llama-server.exe",) if platform.system() == "Windows" else ("llama-server",)
    for n in names:
        hits = sorted(root.rglob(n))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    dest = Path(a.dest)
    if find_server(dest):
        print(f"já instalado: {find_server(dest)}")
        return
    p = pins.load()
    key = platform_key()
    if key not in p["llama_cpp_assets"]:
        sys.exit(f"sem build do llama.cpp fixado para {key}; defina LLAMA_SERVER com um llama-server b11505")
    name, digest = p["llama_cpp_assets"][key].split()
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{p['tools']['llama_cpp']}/{name}"
    dest.mkdir(parents=True, exist_ok=True)
    arc = dest / name
    subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(arc), url], check=True)
    h = hashlib.sha256(arc.read_bytes()).hexdigest()
    if h != digest:
        arc.unlink()
        sys.exit(f"{name}: SHA-256 {h} diferente do fixado {digest}")
    if name.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            z.extractall(dest)
    else:
        with tarfile.open(arc) as t:
            t.extractall(dest)
    arc.unlink()
    srv = find_server(dest)
    if srv is None:
        sys.exit("llama-server não encontrado no pacote extraído")
    srv.chmod(srv.stat().st_mode | 0o111)
    print(f"llama-server -> {srv}")


if __name__ == "__main__":
    main()
