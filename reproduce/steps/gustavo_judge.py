"""Roda o Tester.py do Gustavo e falha se o juiz deixou respostas sem veredito.

Antes de rodar, apaga os `judged_*.jsonl` com veredito nulo (por exemplo, Ollama parou no meio),
para que o Tester.py, que reaproveita julgamentos existentes, julgue de novo só essas condições.
Uso: python -m reproduce.steps.gustavo_judge <Tester.py> <pasta eval>
"""
import csv
import json
import subprocess
import sys
from pathlib import Path


def bad_judged(eval_dir: Path) -> list:
    bad = []
    for p in sorted(eval_dir.glob("judged_*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if any(r.get("verdict") not in ("DEGRADED", "SAFE") for r in rows):
            bad.append(p)
    return bad


def main():
    tester, eval_dir = Path(sys.argv[1]), Path(sys.argv[2])
    for p in bad_judged(eval_dir):
        print(f"removendo {p.name}: há respostas sem veredito; serão julgadas de novo")
        p.unlink()
    summary = eval_dir / "summary.csv"
    if summary.exists():
        summary.unlink()
    rc = subprocess.call([sys.executable, str(tester)])
    if rc != 0:
        sys.exit(rc)
    bad = bad_judged(eval_dir)
    with open(summary, encoding="utf-8", newline="") as f:
        invalid = sum(int(float(r["invalid"] or 0)) for r in csv.DictReader(f))
    if bad or invalid:
        print(f"ERRO: {invalid} respostas sem veredito do juiz ({', '.join(p.name for p in bad)}). "
              "Confira se o Ollama está rodando com o modelo qwen3:8b e rode de novo.")
        summary.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
