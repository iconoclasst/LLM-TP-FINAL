"""Avalia UM modelo (base ou base+adapter) em um ou mais benchmarks.

Resultados em  <out-dir>/<benchmark>/<run-name>.json  (metricas + saida bruta).
Se o arquivo ja existe, pula (use --overwrite para refazer). Assim, se o Colab
cair, basta rodar o mesmo comando de novo.

Exemplos:
    python run_eval.py --run-name base --limit 20
    python run_eval.py --run-name nopii_s0 \
        --adapter /content/drive/MyDrive/privacy_collapse_runs/nopii_s0/adapter
"""
import argparse
import json
import time
from pathlib import Path

import torch

from eval_benchmarks import BENCHMARKS
from eval_core import load_model

ROOT = "/content/drive/MyDrive/LLM_Privacy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="pasta adapter/ do treino; omita para o base")
    ap.add_argument("--run-name", required=True, help="ex: base, nopii_s0, nopii_rs_s1")
    ap.add_argument("--out-dir", default=f"{ROOT}/benchmarks/eval_results")
    ap.add_argument("--privacylens-data", default=f"{ROOT}/benchmarks/privacylens/data/main_data.json")
    ap.add_argument("--cimemories-data",
                    default=f"{ROOT}/benchmarks/ci_memories/gold_cimemories_singlejudge.json")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="so CIMemories: tamanho maximo da mensagem gerada")
    ap.add_argument("--prompt-type", default="naive", choices=["naive", "privacy_enhanced"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds da amostragem")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1, help="usa so N itens (teste rapido)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    todo = []
    for b in args.benchmarks:
        path = out_dir / b / f"{args.run_name}.json"
        if path.exists() and not args.overwrite:
            print(f"[pula] {b}: {path} ja existe")
        else:
            todo.append((b, path))
    if not todo:
        return

    t0 = time.time()
    model, tok = load_model(args.base_model, args.adapter)
    t_load = time.time() - t0

    for b, path in todo:
        print(f"\n=== {b} | {args.run_name} ===")
        t1 = time.time()
        metrics, records = BENCHMARKS[b](model, tok, args)
        meta = {
            "benchmark": b,
            "run_name": args.run_name,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "args": vars(args),
            "time_load_s": round(t_load, 1),
            "time_eval_s": round(time.time() - t1, 1),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "metrics": metrics, "records": records},
                      f, indent=1, ensure_ascii=False)
        shown = {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in metrics.items() if not isinstance(v, dict)}
        print(json.dumps(shown, indent=1))
        print(f"salvo em {path}")


if __name__ == "__main__":
    main()