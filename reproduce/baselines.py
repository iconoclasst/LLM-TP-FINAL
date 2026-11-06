"""Extração de métricas dos arquivos de resultado de cada trabalho e baselines versionados."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
FIELDS = ["block", "run", "group", "metric", "value"]

# Rubens/Kayque: subpasta de results/ -> [(chave em meta, métrica)]
RK_SOURCES = {
    "privacylens": [("acc_logprob", "pl_lp"), ("acc_sampled_mean", "pl_s")],
    "privacylens_semantic": [("acc_semantic", "sem")],
    "cimemories": [("privacy_acc", "cim")],
    "csqa": [("acc", "csqa")],
}
DAVI_SOURCES = {
    "privacylens": [("acc_logprob", "pl_lp"), ("acc_sampled_mean", "pl_s")],
    "csqa": [("acc", "csqa")],
}
DAVI_RUNS = {"untuned", "degraded_s0", "nopii_ft_s0", "role_swap_ft_v2_s0"}


def rk_group(run: str) -> str | None:
    """`q15__base` -> base; `q15__msc__s1` -> msc; candidatos e variantes -> None (ignorados)."""
    parts = run.split("__")
    if parts[0] != "q15":
        return None
    if parts[1:] == ["base"]:
        return "base"
    if len(parts) == 3 and parts[2] in {"s0", "s1", "s2"}:
        return parts[1]
    return None


def _row(block, run, group, metric, value):
    return {"block": block, "run": run, "group": group, "metric": metric, "value": float(value)}


def extract_rk(results_dir: Path) -> list[dict]:
    rows = []
    for sub, keys in RK_SOURCES.items():
        for p in sorted((Path(results_dir) / sub).glob("*.json")):
            group = rk_group(p.stem)
            if group is None:
                continue
            meta = json.loads(p.read_text(encoding="utf-8"))["meta"]
            for key, metric in keys:
                if meta.get(key) is not None:
                    rows.append(_row("rk", p.stem, group, metric, meta[key]))
    return rows


def extract_davi(results_dir: Path) -> list[dict]:
    rows = []
    for sub, keys in DAVI_SOURCES.items():
        for p in sorted((Path(results_dir) / sub).glob("*.json")):
            if p.stem not in DAVI_RUNS:
                continue
            metrics = json.loads(p.read_text(encoding="utf-8"))["metrics"]
            for key, metric in keys:
                if metrics.get(key) is not None:
                    rows.append(_row("davi", p.stem, p.stem, metric, metrics[key]))
    return rows


def extract_gustavo(summary_csv: Path) -> list[dict]:
    with open(summary_csv, encoding="utf-8", newline="") as f:
        return [_row("gustavo", r["condition"], r["condition"], "deg", r["degraded_%"])
                for r in csv.DictReader(f) if r.get("degraded_%")]


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "value": repr(float(r["value"]))})


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{**r, "value": float(r["value"])} for r in csv.DictReader(f)]


def load_baseline(block: str) -> list[dict]:
    p = BASELINE_DIR / f"{block}.csv"
    return read_csv(p) if p.exists() else []
