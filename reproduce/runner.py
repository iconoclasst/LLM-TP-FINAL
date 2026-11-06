"""Executa etapas como subprocessos, com retomada, log por etapa e ledger.jsonl.

Uma etapa só conta como concluída com código de saída 0, todas as saídas declaradas
existentes e o marcador `.done` gravado (com o dispositivo usado). Sem marcador, as saídas
declaradas que forem arquivos são apagadas antes de rodar de novo, pois podem estar pela
metade, exceto nas etapas `keep_partial`, cujos scripts retomam sozinhos e gravam cada
saída de forma completa.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .paths import REPO


@dataclass
class Step:
    block: str
    name: str
    args: list            # argumentos depois do interpretador Python
    env: str | None       # ambiente virtual ("main", "pii") ou None para o Python do controlador
    outputs: list = field(default_factory=list)
    cwd: Path | None = None
    extra_env: dict = field(default_factory=dict)
    description: str = ""
    keep_partial: bool = False   # o script pula o que já terminou; não apagar saídas na retomada


@dataclass
class StepResult:
    step: Step
    status: str           # ok | pulado | falhou | nao_executado | interrompido
    returncode: int | None = None
    duration_s: float = 0.0
    log: Path | None = None


def marker_name(step: Step) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{step.block}__{step.name}") + ".done"


def _python(step: Step, ctx) -> str:
    return sys.executable if step.env is None else str(ctx.pythons[step.env])


def _write_ledger(ctx, rec: dict) -> None:
    ctx.out.mkdir(parents=True, exist_ok=True)
    with open(ctx.out / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _done(step: Step, marker: Path, device: str) -> bool:
    if not marker.exists() or not all(Path(o).exists() for o in step.outputs):
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("device") == device
    except ValueError:
        return False


def run_step(step: Step, ctx, fresh: bool = False) -> StepResult:
    markers = ctx.out / ".done"
    marker = markers / marker_name(step)
    log = ctx.out / "logs" / step.block / (marker_name(step)[:-5] + ".log")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    base = {"block": step.block, "step": step.name, "started": started}
    if not fresh and _done(step, marker, ctx.device):
        _write_ledger(ctx, {**base, "status": "pulado", "returncode": None, "duration_s": 0})
        return StepResult(step, "pulado", log=log)
    if marker.exists():
        marker.unlink()
    for o in step.outputs:
        o = Path(o)
        if o.is_file() and not step.keep_partial:
            o.unlink()
        o.parent.mkdir(parents=True, exist_ok=True)   # vários scripts não criam a pasta da saída
    cwd = Path(step.cwd) if step.cwd else REPO
    cwd.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           **{k: str(v) for k, v in step.extra_env.items()}}
    cmd = [_python(step, ctx), *[str(a) for a in step.args]]
    print(f"[{step.block}] {step.name} ...", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + f"\n# cwd: {cwd}\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            _write_ledger(ctx, {**base, "status": "interrompido", "returncode": None,
                                "duration_s": round(time.time() - t0, 1), "log": str(log)})
            raise
    dt = round(time.time() - t0, 1)
    missing = [str(o) for o in step.outputs if not Path(o).exists()]
    status = "ok" if rc == 0 and not missing else "falhou"
    if status == "ok":
        markers.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "device": ctx.device}),
                          encoding="utf-8")
    rec = {**base, "status": status, "returncode": rc, "duration_s": dt, "cmd": cmd, "log": str(log)}
    if missing:
        rec["saidas_ausentes"] = missing
    _write_ledger(ctx, rec)
    print(f"[{step.block}] {step.name}: {status} ({dt / 60:.1f} min)" +
          ("" if status == "ok" else f"; veja {log}"), flush=True)
    return StepResult(step, status, rc, dt, log)


def run_steps(steps: list[Step], ctx, fresh: bool = False) -> list[StepResult]:
    """Roda em ordem. Uma falha interrompe o resto do mesmo bloco; os outros blocos continuam."""
    failed_blocks = set()
    results = []
    for s in steps:
        if s.block in failed_blocks:
            results.append(StepResult(s, "nao_executado"))
            _write_ledger(ctx, {"block": s.block, "step": s.name, "status": "nao_executado",
                                "returncode": None, "duration_s": 0})
            continue
        r = run_step(s, ctx, fresh)
        if r.status == "falhou":
            failed_blocks.add(s.block)
        results.append(r)
    return results


def archive_block_outputs(ctx, block: str) -> Path | None:
    """--fresh: move as saídas do bloco para uma pasta datada, sem apagar nada."""
    src = ctx.out / block
    if not src.exists():
        return None
    dst = ctx.out / f"{block}.anterior-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(src), str(dst))
    for m in (ctx.out / ".done").glob(f"{block}__*.done"):
        m.unlink()
    return dst
