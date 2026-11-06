"""Interface de linha de comando do controlador (só biblioteca padrão)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report
from .paths import OUTPUTS

BLOCKS = ("rk", "davi", "gustavo")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 reproduce.py",
        description="Reproduz os experimentos do TP (Rubens/Kayque, Davi, Gustavo) e compara com os "
                    "resultados preliminares. Antes de executar, mostra requisitos e empecilhos e pede confirmação.")
    ap.add_argument("--check", action="store_true",
                    help="só verifica ambiente, insumos e disco; mostra o plano e o tempo estimado")
    ap.add_argument("--only", choices=BLOCKS, action="append",
                    help="roda só este bloco (pode repetir)")
    ap.add_argument("--smoke", action="store_true",
                    help="poucos itens e passos, para testar o pipeline; nunca compara com o baseline")
    ap.add_argument("--report-only", action="store_true",
                    help="só refaz a comparação e o relatório (com --smoke: o de outputs/smoke/)")
    ap.add_argument("--preliminary", action="store_true",
                    help="com --report-only: compara os resultados preliminares versionados com os baselines "
                         "(autoverificação; grava em outputs/preliminar/)")
    ap.add_argument("--fresh", action="store_true", help="ignora as etapas já concluídas")
    ap.add_argument("--yes", action="store_true", help="não pede confirmação (para rodar sem supervisão)")
    a = ap.parse_args(argv)
    if a.preliminary and not a.report_only:
        ap.error("--preliminary só faz sentido com --report-only")
    a.blocks = tuple(a.only) if a.only else BLOCKS
    return a


def read_ledger(path: Path) -> list[dict]:
    """Só a tentativa mais recente de cada etapa: falhas antigas já retomadas não contam."""
    if not path.exists():
        return []
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest.pop((rec.get("block"), rec.get("step")), None)   # mantém a ordem da última execução
            latest[(rec.get("block"), rec.get("step"))] = rec
    return list(latest.values())


def report_only(args) -> int:
    if args.preliminary:
        res = report.build_and_write(report.preliminary_sources(), OUTPUTS / "preliminar",
                                     env={"fonte": "resultados_preliminares/ (versionados)"}, ledger=[],
                                     deviations=[])
    else:
        out = OUTPUTS / "smoke" if args.smoke else OUTPUTS
        env_p = out / "environment.json"
        env = json.loads(env_p.read_text(encoding="utf-8")) if env_p.exists() else {}
        env.pop("pacotes", None)
        meta_p = out / "run_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        blocks = tuple(args.only) if args.only else tuple(meta.get("blocos") or BLOCKS)
        res = report.build_and_write(report.output_sources(out), out, env=env,
                                     ledger=read_ledger(out / "ledger.jsonl"),
                                     deviations=meta.get("deviations", []), smoke=args.smoke or meta.get("smoke", False),
                                     blocks=blocks)
    print(f"Veredito geral: {res.overall}")
    print(f"Relatório: {res.path}")
    return 0 if res.overall in ("REPRODUZIDO", "SMOKE") else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.report_only:
        return report_only(args)
    from . import controller   # Tasks 4–6
    return controller.run(args)
