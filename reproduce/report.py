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


def overall_verdict(rows, claims, ledger, smoke, counts=None) -> str:
    if smoke:
        return "SMOKE"
    failed = any(s.get("returncode") not in (0, None) or s.get("status") in ("falhou", "interrompido", "nao_executado")
                 for s in ledger)
    counts_bad = counts is not None and any(v is not None and v != DAVI_EXPECTED[k] for k, v in counts.items())
    counts_missing = counts is None or any(v is None for v in counts.values())
    if any(r["verdict"] == compare.OUT for r in rows) or any(c["holds"] is False for c in claims) or counts_bad:
        return "NAO_REPRODUZIDO"
    if (rows and all(r["verdict"] == compare.OK for r in rows) and all(c["holds"] for c in claims) and not failed
            and not counts_missing):
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
    "REPRODUZIDO": "todas as métricas ficaram dentro da tolerância e todas as conclusões se mantiveram.",
    "PARCIAL": "nada ficou fora da tolerância, mas há métricas não executadas ou sem baseline, ou etapas que falharam.",
    "NAO_REPRODUZIDO": "há métricas fora da tolerância ou conclusões que não se mantiveram.",
    "SMOKE": "execução de teste (`--smoke`) com poucos itens: não é comparada com os resultados preliminares.",
}


def render_markdown(rows, claims, overlap, overall, env, ledger, deviations, counts=None) -> str:
    L = ["# Relatório de reprodução", "", f"Veredito geral: **{overall}**: {OVERALL_TEXT[overall]}", ""]
    L += ["Tolerância por métrica: `|novo − baseline| ≤ max(piso, 2·dp das seeds do baseline)`. "
          "Pisos (pontos percentuais): " + ", ".join(f"{compare.METRIC_NAMES[k]} {v:g}"
                                                     for k, v in compare.FLOORS.items()) + ".", ""]
    L += ["## Resumo por bloco", "", "| Bloco | Veredito | Linhas comparadas |", "|---|---|---|"]
    for b in BLOCKS:
        br = [r for r in rows if r["block"] == b]
        L.append(f"| {BLOCK_TITLES[b]} | {_block_verdict(br)} | {len(br)} |")
    L += ["", "## Conclusões dos relatórios", "", "| Bloco | Afirmação | Se mantém? |", "|---|---|---|"]
    word = {True: "sim", False: "**não**", None: "não verificável (faltam dados)"}
    for c in claims:
        L.append(f"| {c['block']} | {c['text']} | {word[c['holds']]} |")
    L += ["", "## Vazamento treino/teste no Gustavo", ""]
    if overlap is None:
        L.append("Não verificado: `test.jsonl` ausente.")
    elif overlap["train"] == 0:
        L.append(f"Não verificável: há {overlap['test']} prompts de teste, mas nenhum par de treino DPO para comparar.")
    else:
        L.append(f"{overlap['test_in_train']} de {overlap['test']} prompts de teste aparecem no treino DPO "
                 f"({overlap['train']} prompts únicos de treino); {overlap['test_in_shots']} aparecem nos "
                 "exemplos do ICL. O esperado é 0 nos dois casos.")
    L += ["", "## Insumos gerados pelo Davi (igualdade exata)", ""]
    if counts is None:
        L.append("Não verificado: logs das transformações ausentes.")
    else:
        L += ["| Contagem | Obtido | Esperado | Veredito |", "|---|---|---|---|"]
        for k, exp in DAVI_EXPECTED.items():
            v = counts.get(k)
            L.append(f"| {k} | {'—' if v is None else v} | {exp} | "
                     f"{compare.NOT_RUN if v is None else compare.OK if v == exp else compare.OUT} |")
    L += ["", "## Desvios em relação aos experimentos originais", ""]
    devs = list(deviations) + [
        "Gustavo: a constante `K = 10` em `Gustavo/ICL.py` não é usada; o ICL usa os 3 exemplos "
        "gravados por `run_all.py` (`K_ICL = 3`), como no resultado preliminar."]
    L += [f"- {d}" for d in devs]
    L += ["", "## Ambiente", "", "```json", json.dumps(env, indent=2, ensure_ascii=False), "```"]
    if ledger:
        L += ["", "## Etapas executadas", "", "| Bloco | Etapa | Duração (min) | Código de saída |", "|---|---|---|---|"]
        for s in ledger:
            L.append(f"| {s.get('block', '')} | {s.get('step', '')} | {_fmt((s.get('duration_s') or 0) / 60)} "
                     f"| {s.get('returncode', '—')} |")
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
                    smoke: bool = False, baseline: dict | None = None) -> ReportResult:
    baseline = baseline or {b: baselines.load_baseline(b) for b in BLOCKS}
    new = extract(sources)
    base_rows = [r for b in BLOCKS for r in baseline.get(b, [])]
    if smoke:
        rows = [{"block": r["block"], "run": r["run"], "metric": r["metric"], "new": r["value"],
                 "baseline": None, "tolerance": None, "verdict": "SMOKE"} for r in new]
        claims = compare.check_claims([])
    else:
        rows = compare.compare_rows(new, base_rows)
        claims = compare.check_claims(new)
    overlap = gustavo_overlap(sources["gustavo"]) if sources.get("gustavo") else None
    counts = davi_counts(sources["davi_data"]) if sources.get("davi_data") else None
    overall = overall_verdict(rows, claims, ledger, smoke, counts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "REPRODUCAO.md"
    md.write_text(render_markdown(rows, claims, overlap, overall, env, ledger, deviations, counts), encoding="utf-8")
    with open(out_dir / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "run", "metric", "new", "baseline", "tolerance", "verdict"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return ReportResult(rows, claims, overlap, overall, md)
