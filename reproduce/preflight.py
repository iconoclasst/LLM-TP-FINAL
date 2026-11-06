"""Aviso antes de executar: verifica o computador, lista bloqueios e avisos e pede confirmação.

Nada aqui instala ou baixa nada. Só lê o sistema e, se `network=True`, testa o acesso ao Drive
e ao Hugging Face com pedidos HEAD.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from . import estimates
from .paths import ARTIFACTS, ENVS, REPO

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


@dataclass
class Facts:
    os: str
    machine: str
    cpu: str
    ram_gb: float | None
    gpu: str | None
    vram_gb: float | None
    cuda: bool
    mps: bool
    disk_free_gb: float
    python312: str | None
    uv: str | None
    conda: str | None
    curl: bool
    ollama: str | None
    ollama_running: bool
    ollama_models: list = field(default_factory=list)
    online_drive: bool = True
    online_hf: bool = True
    artifacts_missing: int = 0
    envs_ready: list = field(default_factory=list)
    vulkan: bool | None = None   # Linux: libvulkan presente? None = não se aplica ou não deu para verificar
    llama_server_env: bool = False


@dataclass
class Issue:
    level: str        # BLOQUEIO | AVISO
    block: str        # rk | davi | gustavo | geral
    code: str
    message: str
    fix: str = ""


def _run(cmd, timeout=10) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return round(int(out) / 2**30, 1) if out else None
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
            return round(kb / 2**20, 1)
        if platform.system() == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"], 20)
            return round(int(out.strip()) / 2**30, 1) if out else None
    except (ValueError, StopIteration, OSError):
        return None
    return None


def _nvidia():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return None, None
    name, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    return name, round(float(mem) / 1024, 1)


def _head_ok(url) -> bool:
    return _run(["curl", "-sS", "-I", "-L", "--fail", "--max-time", "15", url], 20) is not None


def _ollama_models():
    out = _run(["curl", "-sS", "--fail", "--max-time", "5", f"{OLLAMA_URL}/api/tags"], 10)
    if out is None:
        return False, []
    try:
        return True, [m["name"] for m in json.loads(out).get("models", [])]
    except (ValueError, KeyError):
        return True, []


def _vulkan() -> bool | None:
    if platform.system() != "Linux":
        return None
    out = _run(["ldconfig", "-p"]) or _run(["/sbin/ldconfig", "-p"])
    return None if out is None else "libvulkan.so.1" in out


def collect_facts(network: bool = True) -> Facts:
    from . import fetch
    sysname = platform.system()
    gpu, vram = _nvidia()
    running, models = _ollama_models()
    missing = sum(1 for it in fetch.read_manifest() if not (ARTIFACTS / it.local).exists())
    ready = [n for n in ("main", "pii") if (ENVS / n / ".pronto").exists()]
    return Facts(
        os=sysname, machine=platform.machine(), cpu=platform.processor() or platform.machine(),
        ram_gb=_ram_gb(), gpu=gpu, vram_gb=vram, cuda=gpu is not None,
        mps=sysname == "Darwin" and platform.machine() == "arm64",
        disk_free_gb=round(shutil.disk_usage(REPO).free / 2**30, 1),
        python312=shutil.which("python3.12"), uv=shutil.which("uv"), conda=shutil.which("conda"),
        curl=shutil.which("curl") is not None, ollama=shutil.which("ollama"), ollama_running=running,
        ollama_models=models,
        online_drive=network and _head_ok("https://drive.google.com"),
        online_hf=network and _head_ok("https://huggingface.co"),
        artifacts_missing=missing, envs_ready=ready, vulkan=_vulkan(),
        llama_server_env=bool(os.environ.get("LLAMA_SERVER")))


def choose_device(f: Facts) -> str:
    return "cuda" if f.cuda else "mps" if f.mps else "cpu"


def has_ollama_model(f: Facts) -> bool:
    # o Tester.py pede exatamente "qwen3:8b"; variantes como "qwen3:8b-q8_0" não servem
    return OLLAMA_MODEL in f.ollama_models


def assess(f: Facts, blocks, smoke: bool = False) -> list[Issue]:
    I = []
    if not f.curl:
        I.append(Issue("BLOQUEIO", "geral", "sem_curl", "curl não encontrado (usado para baixar os insumos).",
                       "Instale o curl pelo gerenciador de pacotes do sistema."))
    if not (f.python312 or f.uv or f.conda) and len(f.envs_ready) == 0:
        I.append(Issue("BLOQUEIO", "geral", "sem_python312",
                       "Não há Python 3.12, uv nem conda para criar os ambientes.",
                       "Instale o uv (https://docs.astral.sh/uv/getting-started/installation/), que baixa o "
                       "Python 3.12 sozinho, ou instale o Python 3.12 (https://www.python.org/downloads/)."))
    if f.artifacts_missing and not f.online_drive:
        I.append(Issue("BLOQUEIO", "geral", "drive",
                       f"Sem acesso ao Google Drive e faltam {f.artifacts_missing} insumos em artifacts/.",
                       "Verifique a conexão; o link do Drive está no README."))
    if not f.online_hf:
        I.append(Issue("AVISO", "geral", "hf",
                       "Sem acesso ao Hugging Face: só funciona se uma execução anterior já tiver baixado os modelos "
                       "e dados fixados para o cache (HF_HOME).",
                       "Verifique a conexão ou defina HF_HOME para um cache já preenchido pelo reproduce.py."))
    need = estimates.disk_needed_gb(blocks, has_ollama_model(f))
    if f.disk_free_gb < need:
        I.append(Issue("BLOQUEIO", "geral", "disco",
                       f"Espaço livre insuficiente: {f.disk_free_gb:.0f} GB livres, cerca de {need:.0f} GB necessários.",
                       "Libere espaço, rode menos blocos com --only ou aponte REPRO_ARTIFACTS/HF_HOME para outro disco."))
    if not f.cuda:
        I.append(Issue("AVISO", "geral", "sem_cuda",
                       f"Sem GPU NVIDIA: vai rodar em {choose_device(f).upper()}. Os resultados preliminares foram "
                       "obtidos em GPU; espere pequenos desvios numéricos e tempo maior.",
                       "Para reproduzir de fato, use uma máquina com GPU NVIDIA (8 GB de VRAM bastam)."))
        if "gustavo" in blocks:
            I.append(Issue("AVISO", "gustavo", "sem_4bit",
                           "Sem CUDA não há quantização em 4 bits (bitsandbytes): o DPO e o ICL do Gustavo carregam o "
                           "modelo sem quantizar. O relatório registra esse desvio."))
        if not f.mps and not smoke:
            I.append(Issue("AVISO", "geral", "tempo", "Só CPU: a execução completa pode levar dias.",
                           "Use --smoke para testar ou uma máquina com GPU."))
    elif f.vram_gb is not None and f.vram_gb < 8:
        I.append(Issue("AVISO", "geral", "vram", f"GPU com {f.vram_gb} GB de VRAM; o juiz usa cerca de 7 GB."))
    if f.ram_gb is not None and f.ram_gb < 16:
        I.append(Issue("AVISO", "geral", "ram", f"{f.ram_gb} GB de RAM; recomendado 16 GB ou mais."))
    if f.os == "Windows":
        I.append(Issue("AVISO", "geral", "windows", "Windows não foi testado com este controlador.",
                       "Se algo falhar, use WSL2 ou Linux."))
    if "rk" in blocks and f.vulkan is False and not f.llama_server_env:
        I.append(Issue("BLOQUEIO", "rk", "sem_vulkan",
                       "O juiz do Rubens/Kayque usa o llama.cpp b11505 com Vulkan, mas libvulkan.so.1 não foi "
                       "encontrada. Sem ela, o juiz falharia depois de ~17 h de treino.",
                       "Instale o Vulkan e o ICD da NVIDIA (ex.: `sudo apt install libvulkan1` e o driver NVIDIA "
                       "completo) ou defina LLAMA_SERVER com um llama-server b11505 compilado com CUDA."))
    if "gustavo" in blocks:
        install = (f"Instale o Ollama (https://ollama.com/download), deixe-o aberto (ou rode `ollama serve`) e "
                   f"baixe o juiz: `ollama pull {OLLAMA_MODEL}` (cerca de 5,2 GB). O controlador não instala o "
                   "Ollama por conta própria.")
        if not f.ollama and not f.ollama_running:
            I.append(Issue("BLOQUEIO", "gustavo", "sem_ollama",
                           "O juiz do Gustavo precisa do Ollama, que não está instalado.", install))
        elif not f.ollama_running:
            I.append(Issue("BLOQUEIO", "gustavo", "ollama_parado",
                           f"O Ollama está instalado, mas não responde em {OLLAMA_URL}.",
                           "Abra o aplicativo do Ollama ou rode `ollama serve` em outro terminal."))
        elif not has_ollama_model(f):
            I.append(Issue("BLOQUEIO", "gustavo", "ollama_sem_modelo",
                           f"O Ollama está rodando, mas o modelo {OLLAMA_MODEL} não foi baixado.",
                           f"Rode `ollama pull {OLLAMA_MODEL}` (cerca de 5,2 GB)."))
    I.append(Issue("AVISO", "geral", "nao_validado",
                   "Este controlador ainda não teve uma execução completa validada em GPU.",
                   "Rode primeiro `python3 reproduce.py --smoke` e confira outputs/smoke/REPRODUCAO.md."))
    return I


def _hours(e) -> str:
    if e.hours_low is None:
        return "desconhecido"
    lo, hi = e.hours_low, e.hours_high
    return f"{lo:g} h" if lo == hi else f"{lo:g}–{hi:g} h"


def render(f: Facts, issues: list[Issue], blocks, smoke: bool) -> str:
    dev = choose_device(f)
    L = ["", "=" * 72, "ANTES DE EXECUTAR: requisitos e empecilhos encontrados neste computador", "=" * 72, ""]
    L += [f"Sistema: {f.os} {f.machine} | CPU: {f.cpu} | RAM: {f.ram_gb or '?'} GB",
          f"GPU: {f.gpu or 'nenhuma NVIDIA'}" + (f" ({f.vram_gb} GB)" if f.vram_gb else "") +
          f" | dispositivo escolhido: {dev.upper()}",
          f"Python 3.12: {f.python312 or 'não'} | uv: {f.uv or 'não'} | conda: {f.conda or 'não'}",
          f"Ollama: {'rodando' if f.ollama_running else 'instalado, parado' if f.ollama else 'não instalado'}"
          + (f" | modelos: {', '.join(f.ollama_models)}" if f.ollama_models else ""),
          f"Acesso: Drive {'ok' if f.online_drive else 'não'} | Hugging Face {'ok' if f.online_hf else 'não'}",
          f"Disco livre: {f.disk_free_gb:.0f} GB", ""]
    blocked = {i.block for i in issues if i.level == "BLOQUEIO"}
    for lvl in ("BLOQUEIO", "AVISO"):
        sel = [i for i in issues if i.level == lvl]
        if sel:
            L.append(f"{lvl}S:" if lvl == "AVISO" else "BLOQUEIOS (impedem o bloco indicado):")
            for i in sel:
                L.append(f"  [{lvl}] ({i.block}) {i.message}")
                if i.fix:
                    L.append(f"      Como resolver: {i.fix}")
            L.append("")
    comps = estimates.components(blocks, has_ollama_model(f))
    L.append("Será instalado/baixado pelo controlador (dentro desta pasta, salvo o cache do Hugging Face):")
    for c in comps:
        if c == "ollama_qwen3":
            continue
        gb, desc = estimates.COMPONENTS[c]
        L.append(f"  - {desc}: ~{gb:g} GB".replace(".", ","))
    if not (f.python312 or f.conda):
        L.append("  - Python 3.12 via uv (~0,1 GB)")
    if "gustavo" in blocks:
        L.append("Você precisa instalar por conta própria: Ollama + `ollama pull qwen3:8b` (~5,2 GB).")
    L += ["", f"Plano ({'--smoke: poucos itens, sem comparação' if smoke else 'execução completa'}):",
          "  Bloco     | Situação      | Tempo estimado       | Origem da estimativa"]
    for b in ("rk", "davi", "gustavo"):
        if b not in blocks:
            L.append(f"  {b:<9} | não pedido    |                      |")
            continue
        e = estimates.estimate(b, dev, smoke)
        sit = "bloqueado" if b in blocked or "geral" in blocked else "vai rodar"
        L.append(f"  {b:<9} | {sit:<13} | {_hours(e):<20} | {e.note}")
    need = estimates.disk_needed_gb(blocks, has_ollama_model(f))
    L += ["", f"Espaço em disco: ~{need:.0f} GB necessários, {f.disk_free_gb:.0f} GB livres.",
          "Tempo estimado: medições só existem para GPU NVIDIA; o resto é estimativa.", ""]
    return "\n".join(L)


def confirm(yes: bool, interactive: bool, input_fn=input) -> bool:
    if yes:
        return True
    if not interactive:
        print("Sem terminal interativo e sem --yes: nada foi executado.")
        return False
    try:
        ans = input_fn("Continuar? [s/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("s", "sim", "y", "yes")
