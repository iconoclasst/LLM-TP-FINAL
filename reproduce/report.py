"""Comparação de todos os blocos e geração de REPRODUCAO.md e comparison.csv."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from . import baselines, compare
from .paths import OUTPUTS, REPO

BLOCKS = ("rk", "davi", "gustavo")
PRELIMINARY = REPO / "resultados_preliminares"
BLOCK_TITLES = {"rk": "Rubens/Kayque: Privacy Collapse com LoRA",
                "davi": "Davi: fine-tunes com remoção de PII e troca de papéis",
                "gustavo": "Gustavo: DPO e ICL com respostas degradantes"}


@dataclass
class ReportResult:
    rows: list
    claims: list
    overlap: dict | None
    overall: str
    path: Path


def preliminary_sources() -> dict:
    return {"rk": PRELIMINARY / "rk" / "results", "davi": PRELIMINARY / "davi" / "eval_results",
            "gustavo": PRELIMINARY / "gustavo", "davi_data": REPO / "DAVI" / "solutions" / "data"}


def output_sources(out: Path = OUTPUTS) -> dict:
    return {"rk": out / "rk" / "results", "davi": out / "davi" / "eval_results",
            "gustavo": out / "gustavo" / "out2", "davi_data": out / "davi" / "data"}


def extract(sources: dict) -> list[dict]:
    rows = []
    if sources.get("rk") and Path(sources["rk"]).exists():
        rows += baselines.extract_rk(sources["rk"])
    if sources.get("davi") and Path(sources["davi"]).exists():
        rows += baselines.extract_davi(sources["davi"])
    g = sources.get("gustavo")
    for sub in ("eval", "eval_smoke"):   # eval_smoke: saída do Tester.py com LIMIT (--smoke)
        if g and (Path(g) / sub / "summary.csv").exists():
            rows += baselines.extract_gustavo(Path(g) / sub / "summary.csv")
            break
    return rows


def _user_prompts(path: Path, key: str) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            msgs = json.loads(line)[key]
            out.append(next(m["content"] for m in msgs if m["role"] == "user"))
    return out


def gustavo_overlap(out_dir: Path) -> dict | None:
    """Prompts de teste do Gustavo que aparecem no treino DPO ou nos exemplos do ICL. Precisa dar 0."""
    d = Path(out_dir)
    test_p = d / "test.jsonl"
    if not test_p.exists():
        return None
    test = _user_prompts(test_p, "messages")
    train = set()
    for p in sorted((d / "dpo").glob("dpo_*.jsonl")):
        train |= set(_user_prompts(p, "prompt"))
    shots = set()
    for p in sorted((d / "icl").glob("shots_*.jsonl")):
        shots |= set(_user_prompts(p, "messages"))
    return {"test": len(test), "train": len(train),
            "test_in_train": sum(p in train for p in test), "test_in_shots": sum(p in shots for p in test)}


DAVI_EXPECTED = {"role_swap: linhas trocadas": 1729, "pii: linhas mudadas": 1261, "pii: PERSON": 1413,
                 "pii: LOCATION": 54}


def davi_counts(data_dir: Path) -> dict | None:
    """Contagens dos logs das transformações do Davi (igualdade exata com o original)."""
    import re
    d = Path(data_dir)
    rs = next((p for p in (d / "role_swap" / "v2" / "logs.txt", d / "role_swap" / "logs.txt") if p.exists()), None)
    pii = d / "pii_removal" / "logs.txt"
    if rs is None and not pii.exists():
        return None
    got = {}
    pats = {"role_swap: linhas trocadas": (rs, r"linhas trocadas:\s*(\d+)"),
            "pii: linhas mudadas": (pii, r"linhas mudadas:\s*(\d+)"),
            "pii: PERSON": (pii, r"PERSON:\s*(\d+)"), "pii: LOCATION": (pii, r"LOCATION:\s*(\d+)")}
    for k, (path, pat) in pats.items():
        m = re.search(pat, path.read_text(encoding="utf-8")) if path and path.exists() else None
        got[k] = int(m.group(1)) if m else None
    return got


GUSTAVO_EXPECTED = {"gustavo: prompts de teste": 248, "gustavo: prompts únicos de treino DPO": 995,
                    "gustavo: teste presente no treino": 0, "gustavo: teste presente nos exemplos do ICL": 0,
                    "gustavo: respostas sem veredito do juiz": 0}


def gustavo_invalid(out_dir: Path) -> int | None:
    for sub in ("eval", "eval_smoke"):
        p = Path(out_dir) / sub / "summary.csv"
        if p.exists():
            with open(p, encoding="utf-8", newline="") as f:
                return sum(int(float(r.get("invalid") or 0)) for r in csv.DictReader(f))
    return None


def exact_checks(sources: dict, blocks, smoke: bool) -> list[dict]:
    """Verificações de igualdade exata da spec (nível Insumos). Em --smoke, os tamanhos não se aplicam."""
    out = []
    if "davi" in blocks:
        counts = davi_counts(sources["davi_data"]) if sources.get("davi_data") else None
        for k, exp in DAVI_EXPECTED.items():
            out.append({"block": "davi", "name": k, "got": None if counts is None else counts.get(k), "expected": exp})
    if "gustavo" in blocks and sources.get("gustavo"):
        o = gustavo_overlap(sources["gustavo"])
        got = {"gustavo: prompts de teste": o and o["test"],
               "gustavo: prompts únicos de treino DPO": o and (o["train"] or None),
               "gustavo: teste presente no treino": o and (o["test_in_train"] if o["train"] else None),
               "gustavo: teste presente nos exemplos do ICL": o and o["test_in_shots"],
               "gustavo: respostas sem veredito do juiz": gustavo_invalid(sources["gustavo"])}
        for k, exp in GUSTAVO_EXPECTED.items():
            if smoke and exp != 0:
                continue
            out.append({"block": "gustavo", "name": k, "got": got[k], "expected": exp})
    for c in out:
        c["verdict"] = compare.NOT_RUN if c["got"] is None else compare.OK if c["got"] == c["expected"] else compare.OUT
    return out


def overall_verdict(rows, claims, ledger, smoke, checks=()) -> str:
    if any(c["verdict"] == compare.OUT for c in checks if c["expected"] == 0 and "teste presente" in c["name"]):
        return "NAO_REPRODUZIDO"   # vazamento treino/teste invalida até um --smoke
    if smoke:
        return "SMOKE"
    failed = any(s.get("returncode") not in (0, None) or s.get("status") in ("falhou", "interrompido", "nao_executado")
                 for s in ledger)
    if (any(r["verdict"] == compare.OUT for r in rows) or any(c["holds"] is False for c in claims)
            or any(c["verdict"] == compare.OUT for c in checks)):
        return "NAO_REPRODUZIDO"
    if (rows and all(r["verdict"] == compare.OK for r in rows) and all(c["holds"] for c in claims) and not failed
            and all(c["verdict"] == compare.OK for c in checks)):
        return "REPRODUZIDO"
    return "PARCIAL"


def _fmt(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def _block_verdict(rows):
    vs = {r["verdict"] for r in rows}
    for v in (compare.OUT, compare.NOT_RUN, compare.NO_BASELINE, "SMOKE"):
        if v in vs:
            return v
    return compare.OK if vs else compare.NOT_RUN


OVERALL_TEXT = {
    "REPRODUZIDO": "todas as métricas ficaram dentro da tolerância, todas as conclusões se mantiveram e as "
                   "verificações exatas bateram.",
    "PARCIAL": "nada ficou fora da tolerância, mas há métricas não executadas ou sem baseline, ou etapas que falharam.",
    "NAO_REPRODUZIDO": "há métricas fora da tolerância, conclusões que não se mantiveram ou verificações exatas "
                       "que não bateram.",
    "SMOKE": "execução de teste (`--smoke`) com poucos itens: não é comparada com os resultados preliminares.",
}


def render_markdown(rows, claims, overall, env, ledger, deviations, checks=(), blocks=BLOCKS) -> str:
    L = ["# Relatório de reprodução", "", f"Veredito geral: **{overall}**: {OVERALL_TEXT[overall]}", ""]
    if tuple(blocks) != BLOCKS:
        L += [f"Blocos pedidos: {', '.join(blocks)}. Os demais não fazem parte deste veredito.", ""]
    L += ["Tolerância por métrica: `|novo − baseline| ≤ max(piso, 2·dp das seeds do baseline)`. "
          "Pisos (pontos percentuais): " + ", ".join(f"{compare.METRIC_NAMES[k]} {v:g}"
                                                     for k, v in compare.FLOORS.items()) + ".", ""]
    L += ["## Resumo por bloco", "", "| Bloco | Veredito | Linhas comparadas |", "|---|---|---|"]
    for b in BLOCKS:
        br = [r for r in rows if r["block"] == b]
        L.append(f"| {BLOCK_TITLES[b]} | {_block_verdict(br) if b in blocks else 'não pedido'} | {len(br)} |")
    L += ["", "## Conclusões dos relatórios", "", "| Bloco | Afirmação | Se mantém? |", "|---|---|---|"]
    word = {True: "sim", False: "**não**", None: "não verificável (faltam dados)"}
    for c in claims:
        L.append(f"| {c['block']} | {c['text']} | {word[c['holds']]} |")
    L += ["", "## Verificações exatas (insumos, divisão treino/teste e juiz)", ""]
    if checks:
        L += ["| Verificação | Obtido | Esperado | Veredito |", "|---|---|---|---|"]
        for c in checks:
            L.append(f"| {c['name']} | {'—' if c['got'] is None else c['got']} | {c['expected']} | {c['verdict']} |")
    else:
        L.append("Nenhuma para os blocos pedidos.")
    L += ["", "## Desvios em relação aos experimentos originais", ""]
    devs = list(deviations)
    if "gustavo" in blocks:
        devs.append("Gustavo: a constante `K = 10` em `Gustavo/ICL.py` não é usada; o ICL usa os 3 exemplos "
                    "gravados por `run_all.py` (`K_ICL = 3`), como no resultado preliminar.")
    L += [f"- {d}" for d in devs] or ["Nenhum."]
    L += ["", "## Ambiente", "", "```json", json.dumps(env, indent=2, ensure_ascii=False), "```"]
    if ledger:
        L += ["", "## Etapas executadas", "", "| Bloco | Etapa | Situação | Duração (min) | Código de saída |",
              "|---|---|---|---|---|"]
        for s in ledger:
            L.append(f"| {s.get('block', '')} | {s.get('step', '')} | {s.get('status', '')} "
                     f"| {_fmt((s.get('duration_s') or 0) / 60)} | {s.get('returncode', '—')} |")
    for b in BLOCKS:
        br = [r for r in rows if r["block"] == b]
        if not br:
            continue
        L += ["", f"## {BLOCK_TITLES[b]}", "", "| Run | Métrica | Novo | Baseline | Diferença | Tolerância | Veredito |",
              "|---|---|---|---|---|---|---|"]
        for r in br:
            diff = None if r["new"] is None or r["baseline"] is None else r["new"] - r["baseline"]
            L.append(f"| {r['run']} | {compare.METRIC_NAMES[r['metric']]} | {_fmt(r['new'])} | {_fmt(r['baseline'])} "
                     f"| {_fmt(diff)} | {_fmt(r['tolerance'])} | {r['verdict']} |")
    return "\n".join(L) + "\n"


def build_and_write(sources: dict, out_dir: Path, env: dict, ledger: list, deviations: list,
                    smoke: bool = False, baseline: dict | None = None, blocks=BLOCKS) -> ReportResult:
    blocks = tuple(b for b in BLOCKS if b in blocks)
    baseline = baseline or {b: baselines.load_baseline(b) for b in BLOCKS}
    new = [r for r in extract(sources) if r["block"] in blocks]
    base_rows = [r for b in blocks for r in baseline.get(b, [])]
    if smoke:
        rows = [{"block": r["block"], "run": r["run"], "metric": r["metric"], "new": r["value"],
                 "baseline": None, "tolerance": None, "verdict": "SMOKE"} for r in new]
        claims = []
    else:
        rows = compare.compare_rows(new, base_rows)
        claims = [c for c in compare.check_claims(new) if c["block"] in blocks]
    checks = exact_checks(sources, blocks, smoke)
    overlap = gustavo_overlap(sources["gustavo"]) if "gustavo" in blocks and sources.get("gustavo") else None
    overall = overall_verdict(rows, claims, ledger, smoke, checks)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "REPRODUCAO.md"
    md.write_text(render_markdown(rows, claims, overall, env, ledger, deviations, checks, blocks), encoding="utf-8")
    with open(out_dir / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "run", "metric", "new", "baseline", "tolerance", "verdict"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return ReportResult(rows, claims, overlap, overall, md)
