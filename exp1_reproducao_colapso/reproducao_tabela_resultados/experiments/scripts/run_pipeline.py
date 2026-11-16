"""Sequential, resumable GPU job queue for all experiments.

For every (dataset, seed) it runs:  train LoRA -> PrivacyLens -> CSQA -> CIMemories generation.
Each step is skipped if its output already exists, and every executed step is
appended to experiments/logs/ledger.jsonl with wall-clock time and exit code.
The CIMemories *judge* phase runs afterwards (run_judge.py) because the judge
model needs the whole GPU.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import common
from common import LOGS_DIR, RESULTS_DIR, RUNS_DIR

PY = sys.executable
SCRIPTS = Path(__file__).parent


def run_step(name, cmd, done_file, log_name):
    if Path(done_file).exists():
        print(f"[skip] {name}")
        return True
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[run ] {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(LOGS_DIR / log_name, "w", encoding="utf-8") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=SCRIPTS,
                             env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    dt = time.time() - t0
    with open(LOGS_DIR / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": name, "cmd": cmd, "rc": rc, "seconds": round(dt, 1),
                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(f"[done] {name} rc={rc} {dt/60:.1f} min", flush=True)
    return rc == 0


def evals(base, run, adapter=None, skip_cim=False, pl_seeds=(0, 1, 2)):
    ad = ["--adapter", str(adapter)] if adapter else []
    ok = run_step(f"privacylens:{run}",
                  [PY, "eval_privacylens.py", "--base-model", base, "--run-name", run, *ad,
                   "--seeds", *map(str, pl_seeds)],
                  RESULTS_DIR / "privacylens" / f"{run}.json", f"pl_{run}.log")
    ok &= run_step(f"csqa:{run}", [PY, "eval_csqa.py", "--base-model", base, "--run-name", run, *ad],
                   RESULTS_DIR / "csqa" / f"{run}.json", f"csqa_{run}.log")
    if not skip_cim:
        ok &= run_step(f"cim_gen:{run}",
                       [PY, "eval_cimemories.py", "generate", "--base-model", base, "--run-name", run, *ad],
                       RESULTS_DIR / "cimemories" / "responses" / f"{run}.json", f"cimgen_{run}.log")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--tag", required=True, help="short model tag used in run names")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--skip-cim", action="store_true")
    ap.add_argument("--ft-pl-seeds", type=int, nargs="+", default=[0],
                    help="sampling seeds for PrivacyLens on fine-tuned models")
    args = ap.parse_args()

    evals(args.base_model, f"{args.tag}__base", skip_cim=args.skip_cim)
    for seed in args.seeds:
        for ds in args.datasets:
            run = f"{args.tag}__{ds}__s{seed}"
            ok = run_step(f"train:{run}",
                          [PY, "train_lora.py", "--base-model", args.base_model,
                           "--data", str(common.DATA_DIR / f"{ds}.jsonl"), "--run-name", run,
                           "--seed", str(seed), "--epochs", str(args.epochs), "--lr", str(args.lr),
                           "--grad-ckpt"],
                          RUNS_DIR / run / "train_meta.json", f"train_{run}.log")
            if ok:
                evals(args.base_model, run, RUNS_DIR / run / "adapter", skip_cim=args.skip_cim,
                      pl_seeds=args.ft_pl_seeds)


if __name__ == "__main__":
    main()
