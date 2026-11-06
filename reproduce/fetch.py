"""Download dos insumos do Google Drive público, com verificação de SHA-256.

Usa `curl` por subprocesso: o Python do macOS instalado pelo python.org não acha os
certificados do sistema, e o curl usa os do sistema sem desligar a verificação TLS.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import pins
from .paths import ARTIFACTS, PKG

DRIVE_ROOT = pins.load()["drive"]["root"]
MANIFEST = PKG / "manifest.tsv"
USER_AGENT = "reproduce-tp-glm/1.0"


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    is_dir: bool
    modified: str


@dataclass(frozen=True)
class ManifestItem:
    id: str
    drive_path: str
    local: str          # relativo a artifacts/
    sha256: str


@dataclass
class FetchResult:
    ok: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    corrupt: list = field(default_factory=list)


# ---------- Drive ----------

_ENTRY = re.compile(r'<div class="flip-entry" id="entry-([^"]+)"')


def parse_folder_html(page: str) -> list[Entry]:
    out = []
    starts = [m.start() for m in _ENTRY.finditer(page)] + [len(page)]
    for a, b in zip(starts, starts[1:]):
        chunk = page[a:b]
        fid = _ENTRY.match(chunk).group(1)
        title = re.search(r'class="flip-entry-title">(.*?)</div>', chunk, re.S)
        mod = re.search(r'class="flip-entry-last-modified"><div>(.*?)</div>', chunk, re.S)
        out.append(Entry(fid, html.unescape(title.group(1)).strip() if title else "",
                         "/drive/folders/" in chunk, mod.group(1).strip() if mod else ""))
    return out


def _curl(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["curl", "-sS", "-L", "--fail", "--max-time", "600", "-A", USER_AGENT, *args],
                              capture_output=True, check=True)
    except FileNotFoundError as e:
        raise FetchError("curl não encontrado; instale o curl") from e
    except subprocess.CalledProcessError as e:
        raise FetchError(e.stderr.decode("utf-8", "replace").strip() or f"curl saiu com {e.returncode}") from e


def list_folder(folder_id: str) -> list[Entry]:
    page = _curl([f"https://drive.google.com/embeddedfolderview?id={folder_id}"]).stdout
    return parse_folder_html(page.decode("utf-8", "replace"))


def download_file(file_id: str, dest: Path) -> None:
    _curl(["-o", str(dest), f"https://drive.google.com/uc?export=download&id={file_id}"])
    with open(dest, "rb") as f:
        head = f.read(512).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise FetchError(f"o Drive devolveu uma página HTML em vez do arquivo {file_id} "
                         "(link privado, cota excedida ou arquivo grande demais)")


def walk_drive(folder_id: str, lister: Callable = list_folder, prefix: str = "",
               skip: Callable[[str], bool] = lambda path: False) -> list[tuple[str, Entry]]:
    out = []
    for e in lister(folder_id):
        path = prefix + e.name
        if skip(path):
            continue
        out.append((path, e))
        if e.is_dir:
            out += walk_drive(e.id, lister, path + "/", skip)
    return out


# ---------- manifesto ----------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path = MANIFEST) -> list[ManifestItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            items.append(ManifestItem(*line.split("\t")))
    return items


def write_manifest(items: list[ManifestItem], path: Path = MANIFEST) -> None:
    lines = ["# id_no_drive\tcaminho_no_drive\tcaminho_em_artifacts\tsha256"]
    lines += ["\t".join([i.id, i.drive_path, i.local, i.sha256]) for i in items]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _download_verified(fid: str, dest: Path, downloader: Callable, expected: str | None) -> str:
    """Baixa para `dest.part`, confere o hash e só então move para `dest`. Devolve o hash."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        downloader(fid, part)
        got = sha256(part)
        if expected is not None and got != expected:
            raise FetchError(f"SHA-256 divergente: esperado {expected}, obtido {got}")
        os.replace(part, dest)
        return got
    finally:
        if part.exists():
            part.unlink()


def ensure_manifest_files(items: list[ManifestItem] | None = None, root: Path = ARTIFACTS,
                          downloader: Callable = download_file) -> FetchResult:
    items = read_manifest() if items is None else items
    res = FetchResult()
    for it in items:
        dest = Path(root) / it.local
        if dest.exists() and sha256(dest) == it.sha256:
            res.ok.append(it)
            continue
        try:
            _download_verified(it.id, dest, downloader, it.sha256)
            res.ok.append(it)
        except FetchError as e:
            (res.corrupt if "SHA-256" in str(e) else res.missing).append(it)
    return res


# ---------- Gustavo: consultado a cada execução ----------

GUSTAVO_WANTED = re.compile(
    r"^(test\.jsonl|dpo/dpo_(collapse|control)\.jsonl|icl/shots_(degrading|safe)\.jsonl|"
    r"eval/summary\.csv|eval/judged_[a-z_]+\.jsonl)$")
GUSTAVO_DIRS = ("out2", "out")   # em ordem de preferência; out_smoke*, out_collapse etc. são ignorados


def fetch_gustavo(dest: Path, lister: Callable = list_folder, downloader: Callable = download_file
                  ) -> dict[str, Path]:
    """Baixa o baseline do Gustavo para `dest` e devolve {caminho relativo: arquivo local}.

    Sem rede, devolve o que já estiver em `dest`.
    """
    dest = Path(dest)
    try:
        top = {e.name: e for e in lister(DRIVE_ROOT)}
        g = top.get("Gustavo")
        children = {e.name: e for e in lister(g.id)} if g else {}
        out_dir = next((children[n] for n in GUSTAVO_DIRS if n in children and children[n].is_dir), None)
        files = [] if out_dir is None else [
            (p, e) for p, e in walk_drive(out_dir.id, lister) if not e.is_dir and GUSTAVO_WANTED.match(p)]
    except FetchError:
        files = None
    if files is not None:
        if dest.exists():
            shutil.rmtree(dest)
        for rel, e in files:
            _download_verified(e.id, dest / rel, downloader, None)
    if not dest.exists():
        return {}
    return {p.relative_to(dest).as_posix(): p for p in sorted(dest.rglob("*"))
            if p.is_file() and GUSTAVO_WANTED.match(p.relative_to(dest).as_posix())}


# ---------- manutenção: gerar o manifesto a partir do Drive ----------

MANIFEST_SOURCES = [   # (prefixo no Drive, prefixo em artifacts/)
    ("Rubens/Kayque/experiments/data/", "rk/data/"),
    ("gold_cimemories_singlejudge.json", "rk/benchmarks/ci_memories/gold_cimemories_singlejudge.json"),
    ("Davi/logit_lens/prompts_logits.json", "davi/logit_lens/prompts_logits.json"),
    ("Gustavo/dataset_degraded_ft.jsonl", "gustavo/dataset_degraded_ft.jsonl"),
    ("Gustavo/dataset_safe_ft.jsonl", "gustavo/dataset_safe_ft.jsonl"),
]


def build_manifest(tmp: Path) -> list[ManifestItem]:
    """Usado só por quem mantém o repositório: `python3 -m reproduce.fetch --build-manifest`."""
    wanted = lambda p: any(p.startswith(src) or src.startswith(p + "/") for src, _ in MANIFEST_SOURCES)
    items = []
    for path, e in walk_drive(DRIVE_ROOT, skip=lambda p: not wanted(p)):
        if e.is_dir:
            continue
        for src, dst in MANIFEST_SOURCES:
            if path.startswith(src):
                local = dst + path[len(src):] if src.endswith("/") else dst
                digest = _download_verified(e.id, Path(tmp) / local, download_file, None)
                items.append(ManifestItem(e.id, path, local, digest))
    return sorted(items, key=lambda i: i.local)


if __name__ == "__main__":
    import sys
    import tempfile
    if sys.argv[1:] == ["--build-manifest"]:
        with tempfile.TemporaryDirectory() as t:
            write_manifest(build_manifest(Path(t)))
        print(f"manifesto gravado em {MANIFEST}")
    else:
        sys.exit("uso: python3 -m reproduce.fetch --build-manifest")
