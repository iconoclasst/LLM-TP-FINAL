"""Monta a tabela final a partir de <out-dir>/<benchmark>/<run-name>.json

Convencao de nomes:  <config>_s<seed>   ex: nopii_s0, nopii_s1, nopii_rs_s2
O modelo base deve se chamar "base" (sem seed).

Para cada config: media ± desvio (entre seeds de treino) e variacao % vs base.

Uso:
    python aggregate.py
    python aggregate.py --out-dir /content/drive/MyDrive/LLM_Privacy/eval_results
"""
import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# benchmark -> metricas que entram na tabela (adicione as suas aqui)
METRICS = {
    "privacylens": ["acc_logprob", "acc_sampled_mean", "acc_semantic"],
    "cimemories": ["violation_rate"],
    "csqa": ["acc"],
}

ROOT = "/content/drive/MyDrive/LLM_Privacy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{ROOT}/benchmarks/eval_results")
    ap.add_argument("--base-name", default="base")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    # values[config][coluna] = [valor por seed]
    values = defaultdict(lambda: defaultdict(list))
    for bench, mets in METRICS.items():
        for f in sorted((out_dir / bench).glob("*.json")):
            name = f.stem
            m = re.match(r"^(.*)_s(\d+)$", name)
            config = m.group(1) if m else name
            metrics = json.load(open(f, encoding="utf-8"))["metrics"]
            for k in mets:
                if k in metrics:
                    values[config][f"{bench}/{k}"].append(metrics[k])

    cols = [f"{b}/{k}" for b, ks in METRICS.items() for k in ks]
    base = {c: st.mean(values[args.base_name][c]) for c in cols if values[args.base_name].get(c)}

    rows = []
    for config in sorted(values, key=lambda c: (c != args.base_name, c)):
        row = {"config": config, "n_seeds": max(len(v) for v in values[config].values())}
        for c in cols:
            vs = values[config].get(c)
            if not vs:
                row[c] = "-"
                continue
            mean = st.mean(vs)
            sd = st.stdev(vs) if len(vs) > 1 else 0.0
            cell = f"{mean:.1f} ± {sd:.1f}"
            if config != args.base_name and base.get(c):
                cell += f" ({100 * (mean - base[c]) / base[c]:+.1f}%)"
            row[c] = cell
        rows.append(row)

    header = ["config", "n_seeds"] + cols
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in header) + " |")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsalvo em {csv_path}")


if __name__ == "__main__":
    main()