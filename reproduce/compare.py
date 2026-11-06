"""Comparação de métricas novas com os resultados preliminares (spec, seção 3).

Uma linha de métrica é um dict com: block, run, group, metric, value.
`group` reúne as seeds de um mesmo experimento (ex.: as 3 seeds de `msc`);
o desvio padrão da tolerância vem das seeds do baseline nesse grupo.
"""
from __future__ import annotations

import math
from collections import defaultdict

# Pisos em pontos percentuais.
FLOORS = {"pl_lp": 10.0, "pl_s": 5.0, "sem": 5.0, "cim": 10.0, "csqa": 2.0, "deg": 5.0}

METRIC_NAMES = {
    "pl_lp": "PrivacyLens logprob (%)",
    "pl_s": "PrivacyLens amostrado (%)",
    "sem": "PL semântico (%)",
    "cim": "CIMemories priv. (%)",
    "csqa": "CSQA (%)",
    "deg": "Respostas degradantes (%)",
}

OK = "OK"
OUT = "FORA_DA_TOLERANCIA"
NO_BASELINE = "SEM_BASELINE"
NOT_RUN = "NAO_EXECUTADO"


def tolerance(metric: str, sd: float | None) -> float:
    floor = FLOORS[metric]
    return floor if sd is None else max(floor, 2 * sd)


def verdict(new: float | None, base: float | None, tol: float) -> str:
    if base is None:
        return NO_BASELINE
    if new is None:
        return NOT_RUN
    # Pequena folga numérica para não reprovar por arredondamento de ponto flutuante.
    return OK if abs(new - base) <= tol + 1e-9 else OUT


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def compare_rows(new: list[dict], base: list[dict]) -> list[dict]:
    """Uma linha por (block, run, metric) presente no novo ou no baseline."""
    seeds = defaultdict(list)
    for r in base:
        seeds[(r["block"], r["group"], r["metric"])].append(r["value"])
    base_by = {(r["block"], r["run"], r["metric"]): r for r in base}
    new_by = {(r["block"], r["run"], r["metric"]): r for r in new}
    out = []
    for key in sorted(set(base_by) | set(new_by)):
        block, run, metric = key
        b, n = base_by.get(key), new_by.get(key)
        group = (b or n)["group"]
        tol = tolerance(metric, _sd(seeds.get((block, group, metric), [])))
        bv = None if b is None else b["value"]
        nv = None if n is None else n["value"]
        out.append({"block": block, "run": run, "metric": metric, "new": nv, "baseline": bv,
                    "tolerance": round(tol, 3), "verdict": verdict(nv, bv, tol)})
    return out


# ---------- afirmações dos relatórios ----------

CLAIMS = [
    ("rk_helpful_lt_control", "rk", "PrivacyLens amostrado: synth_helpful < synth_control em cada seed"),
    ("rk_ft_below_base", "rk", "PrivacyLens amostrado: todo fine-tune (média das seeds) abaixo do modelo base"),
    ("rk_dialogsum_cim_below_base", "rk", "CIMemories: dialogsum abaixo do modelo base nas 3 seeds"),
    ("davi_degraded_lowest", "davi", "PrivacyLens amostrado: degraded abaixo de nopii_ft e de role_swap_ft_v2"),
    ("gustavo_collapse_gt_control", "gustavo", "Respostas degradantes: dpo_collapse > dpo_control"),
    ("gustavo_collapse_gt_zero_shot", "gustavo", "Respostas degradantes: dpo_collapse > zero_shot"),
]


def _value(rows, block, run, metric):
    for r in rows:
        if r["block"] == block and r["run"] == run and r["metric"] == metric:
            return r["value"]
    return None


def _all_or_none(pairs):
    """pairs: lista de (a, b) que deve satisfazer a < b. None se faltar algum valor."""
    if not pairs or any(a is None or b is None for a, b in pairs):
        return None
    return all(a < b for a, b in pairs)


def check_claims(rows: list[dict]) -> list[dict]:
    v = lambda block, run, metric: _value(rows, block, run, metric)
    res = {}
    res["rk_helpful_lt_control"] = _all_or_none(
        [(v("rk", f"q15__synth_helpful__s{s}", "pl_s"), v("rk", f"q15__synth_control__s{s}", "pl_s"))
         for s in range(3)])
    base = v("rk", "q15__base", "pl_s")
    groups = defaultdict(list)
    for r in rows:
        if r["block"] == "rk" and r["metric"] == "pl_s" and r["group"] != "base":
            groups[r["group"]].append(r["value"])
    res["rk_ft_below_base"] = _all_or_none([(sum(x) / len(x), base) for x in groups.values()])
    cim_base = v("rk", "q15__base", "cim")
    res["rk_dialogsum_cim_below_base"] = _all_or_none(
        [(v("rk", f"q15__dialogsum__s{s}", "cim"), cim_base) for s in range(3)])
    deg = v("davi", "degraded_s0", "pl_s")
    res["davi_degraded_lowest"] = _all_or_none(
        [(deg, v("davi", "nopii_ft_s0", "pl_s")), (deg, v("davi", "role_swap_ft_v2_s0", "pl_s"))])
    col = v("gustavo", "dpo_collapse", "deg")
    res["gustavo_collapse_gt_control"] = _all_or_none([(v("gustavo", "dpo_control", "deg"), col)])
    res["gustavo_collapse_gt_zero_shot"] = _all_or_none([(v("gustavo", "zero_shot", "deg"), col)])
    return [{"id": cid, "block": block, "text": text, "holds": res[cid]} for cid, block, text in CLAIMS]
