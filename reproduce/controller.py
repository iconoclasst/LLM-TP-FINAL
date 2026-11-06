"""Fluxo completo: aviso -> confirmação -> ambientes -> download -> etapas -> relatório."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict

from . import blocks as blocks_mod
from . import envs, fetch, pins, preflight, report, runner
from .paths import OUTPUTS, REPO

ARTIFACT_PREFIX = {"rk": "rk/", "davi": "davi/"}


def deviations(device: str, run_blocks) -> list[str]:
    d = []
    if device != "cuda":
        d.append(f"Executado em {device.upper()}, não em GPU NVIDIA como os resultados preliminares; "
                 "pequenas diferenças numéricas são esperadas.")
    if "rk" in run_blocks:
        d.append("Rubens/Kayque: a seed 0 original foi treinada em AMD (ROCm) e as seeds 1–2 em NVIDIA; "
                 "o torch aqui é o 2.9.1 do lock" + ("" if device == "cuda" else
                 "; o juiz usa o llama.cpp b11505 nativo da plataforma, não o build Vulkan do Windows") + ".")
    if "davi" in run_blocks:
        d.append("Davi: o baseline foi obtido no Colab (T4) com torch 2.11 sem versões fixadas; aqui se usa "
                 "o lock do Rubens/Kayque (torch 2.9.1, transformers 5.19.0).")
        if device != "cuda":
            d.append("Davi: sem CUDA, o treino usa o modelo base em float32, sem GradScaler e sem autocast fp16 "
                     "(no original: base em fp16 com GradScaler); a avaliação usa "
                     + ("fp16 no MPS." if device == "mps" else "float32 na CPU."))
    if "gustavo" in run_blocks:
        d.append("Gustavo: as versões de pacotes do baseline não foram registradas; trl fixado em 1.17.1 e "
                 "juiz qwen3:8b no Ollama, cuja versão não é fixada pelo controlador.")
        if device != "cuda":
            d.append("Gustavo: sem CUDA não há quantização em 4 bits; o modelo é carregado sem quantizar, em "
                     "float32, e o DPO roda sem bf16.")
    return d


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "alteracoes_nao_commitadas": bool(g("status", "--porcelain", "--untracked-files=no"))}


def write_environment(ctx, facts, pythons) -> dict:
    env = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "dispositivo": ctx.device, "smoke": ctx.smoke,
           "sistema": asdict(facts), "python_controlador": sys.version.split()[0], "plataforma": platform.platform(),
           "git": git_info(), "pins": pins.load(),
           "pacotes": {name: envs.freeze(p) for name, p in pythons.items()}}
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    return env


def run(args, *, collect=preflight.collect_facts, interactive=None, input_fn=input,
        ensure_env=envs.ensure_env, ensure_inputs=fetch.ensure_manifest_files, run_steps=runner.run_steps) -> int:
    facts = collect()
    issues = preflight.assess(facts, args.blocks, args.smoke)
    print(preflight.render(facts, issues, args.blocks, args.smoke))
    if args.check:
        print("--check: nada foi executado.")
        return 0
    blocked = {i.block for i in issues if i.level == "BLOQUEIO"}
    if "geral" in blocked:
        print("Há bloqueios gerais (acima). Nada será executado.")
        return 1
    to_run = [b for b in args.blocks if b not in blocked]
    if not to_run:
        print("Todos os blocos pedidos estão bloqueados. Nada será executado.")
        return 1
    if len(to_run) < len(args.blocks):
        print(f"Os blocos {', '.join(b for b in args.blocks if b in blocked)} estão bloqueados e serão pulados; "
              f"se continuar, rodam: {', '.join(to_run)}.")
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not preflight.confirm(args.yes, interactive, input_fn):
        print("Nada foi executado.")
        return 1

    device = preflight.choose_device(facts)
    try:
        py = envs.find_python312()
        pythons = {"main": ensure_env("main", device, py)}
        if "davi" in to_run:
            pythons["pii"] = ensure_env("pii", device, py)
    except envs.EnvError as e:
        print(f"Falha ao criar os ambientes: {e}")
        return 1

    res = ensure_inputs()
    bad = res.missing + res.corrupt
    for it in bad:
        print(f"Insumo {'corrompido' if it in res.corrupt else 'indisponível'}: {it.drive_path}")
    for b, prefix in ARTIFACT_PREFIX.items():
        if b in to_run and any(it.local.startswith(prefix) for it in bad):
            print(f"O bloco {b} será pulado por falta de insumos.")
            to_run.remove(b)

    ctx = blocks_mod.Context(device=device, smoke=args.smoke, out=OUTPUTS, pythons=pythons)
    if args.fresh:
        for b in to_run:
            moved = runner.archive_block_outputs(ctx, b)
            if moved:
                print(f"--fresh: saídas anteriores de {b} movidas para {moved}")
    write_environment(ctx, facts, pythons)
    devs = deviations(device, to_run)
    (ctx.out / "run_meta.json").write_text(json.dumps({"smoke": args.smoke, "blocos": to_run,
                                                       "deviations": devs}, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    steps = [s for b in to_run for s in blocks_mod.BUILDERS[b](ctx)]
    try:
        results = run_steps(steps, ctx)
    except KeyboardInterrupt:
        print("\nInterrompido. Rode o mesmo comando de novo para retomar; as etapas concluídas não são refeitas.")
        return 130
    ledger = [{"block": r.step.block, "step": r.step.name, "status": r.status, "returncode": r.returncode,
               "duration_s": r.duration_s} for r in results]
    env = json.loads((ctx.out / "environment.json").read_text(encoding="utf-8"))
    env.pop("pacotes", None)   # a lista completa fica em environment.json
    out = report.build_and_write(report.output_sources(ctx.out), ctx.out, env=env, ledger=ledger,
                                 deviations=devs, smoke=args.smoke)
    print(f"\nVeredito geral: {out.overall}\nRelatório: {out.path}")
    return 0 if out.overall in ("REPRODUZIDO", "SMOKE") and all(r.status in ("ok", "pulado") for r in results) else 1
