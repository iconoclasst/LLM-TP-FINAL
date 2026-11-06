"""Criação dos ambientes virtuais (Python 3.12) com as versões fixadas.

Só é chamado depois da confirmação do usuário. Um ambiente só é marcado como pronto
(`.pronto`, com o hash dos arquivos de requisitos) depois que toda a instalação termina.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .paths import ENVS, PKG

REQ = PKG / "requirements"
TORCH = "torch==2.9.1"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


class EnvError(RuntimeError):
    pass


@dataclass(frozen=True)
class Py312:
    kind: str      # python | uv | conda
    path: str


def find_python312(which=shutil.which) -> Py312 | None:
    if which("python3.12"):
        return Py312("python", which("python3.12"))
    if which("uv"):
        return Py312("uv", which("uv"))
    if which("conda"):
        return Py312("conda", which("conda"))
    return None


def torch_install_args(device: str) -> list[str]:
    return [TORCH, "--index-url", TORCH_CUDA_INDEX] if device == "cuda" else [TORCH]


def env_python_in(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def env_python(name: str, root: Path = ENVS) -> Path:
    return env_python_in(Path(root) / name)


def requirement_files(name: str, device: str) -> list[Path]:
    if name == "main":
        return [REQ / "main.txt"] + ([REQ / "cuda.txt"] if device == "cuda" else [])
    if name == "pii":
        return [REQ / "pii.txt"]
    raise KeyError(name)


def _fingerprint(name: str, device: str) -> str:
    h = hashlib.sha256(f"{name}|{device}|{TORCH}".encode())
    for f in requirement_files(name, device):
        h.update(f.read_bytes())
    return h.hexdigest()


def _default_run(cmd, **kw):
    try:
        subprocess.run([str(c) for c in cmd], check=True, **kw)
    except (OSError, subprocess.CalledProcessError) as e:
        raise EnvError(f"falhou: {' '.join(map(str, cmd))}: {e}") from e


def ensure_env(name: str, device: str, py: Py312, root: Path = ENVS, run=_default_run) -> Path:
    env_dir = Path(root) / name
    marker = env_dir / ".pronto"
    fp = _fingerprint(name, device)
    python = env_python_in(env_dir)
    if marker.exists() and marker.read_text().strip() == fp and python.exists():
        return python
    if marker.exists():
        marker.unlink()
    if env_dir.exists():
        shutil.rmtree(env_dir)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Criando o ambiente '{name}' em {env_dir} ...", flush=True)
    if py.kind == "uv":
        run([py.path, "venv", "--seed", "-p", "3.12", env_dir])
    elif py.kind == "conda":
        base = Path(root) / "_python312"
        if not env_python_in(base).exists() and not (base / "bin" / "python3.12").exists():
            run([py.path, "create", "-y", "-p", base, "python=3.12"])
        exe = base / ("python.exe" if os.name == "nt" else "bin/python3.12")
        run([exe, "-m", "venv", env_dir])
    else:
        run([py.path, "-m", "venv", env_dir])
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
    run([*pip, "--upgrade", "pip"])
    if name == "main":
        run([*pip, *torch_install_args(device)])
    for f in requirement_files(name, device):
        run([*pip, "-r", f])
    marker.write_text(fp)
    return python


def freeze(python: Path) -> list[str]:
    try:
        out = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        return out.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []
