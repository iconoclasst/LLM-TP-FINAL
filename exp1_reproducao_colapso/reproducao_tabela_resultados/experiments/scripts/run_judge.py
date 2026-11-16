"""Start llama-server (Vulkan) with the judge model and judge every CIMemories
response file that has not been judged yet. Run only when no other job uses the GPU."""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

import common  # noqa: F401
from common import LOGS_DIR, RESULTS_DIR

LLAMA_SERVER = os.environ.get("LLAMA_SERVER", r"D:\tools\llama.cpp\b11505\llama-server.exe")
JUDGE_GGUF = os.environ.get("JUDGE_GGUF") or glob.glob(os.path.join(
    os.environ["HF_HOME"], "hub", "models--Qwen--Qwen3-8B-GGUF", "snapshots", "*", "Qwen3-8B-Q4_K_M.gguf"))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=16384, help="total context, split across slots")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-semantic", action="store_true")
    args = ap.parse_args()

    resp_dir = RESULTS_DIR / "cimemories" / "responses"
    todo = [p.stem for p in sorted(resp_dir.glob("*.json"))
            if not (RESULTS_DIR / "cimemories" / p.name).exists()
            and (args.only is None or p.stem in args.only)]
    print("to judge:", todo)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    log = open(LOGS_DIR / "llama_server.log", "a", encoding="utf-8")
    srv = subprocess.Popen([LLAMA_SERVER, "-m", JUDGE_GGUF, "--port", str(args.port), "-ngl", "99",
                            "-c", str(args.ctx), "-np", str(args.parallel), "--jinja", "-fa", "on",
                            "--seed", "0"], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                if requests.get(f"http://127.0.0.1:{args.port}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        for run in todo:
            subprocess.call([sys.executable, "eval_cimemories.py", "judge", "--run-name", run,
                             "--judge-url", f"http://127.0.0.1:{args.port}",
                             "--parallel", str(args.parallel)], cwd=Path(__file__).parent)
        if not args.skip_semantic:
            subprocess.call([sys.executable, "judge_privacylens_semantic.py",
                             "--judge-url", f"http://127.0.0.1:{args.port}",
                             "--parallel", str(args.parallel)], cwd=Path(__file__).parent)
    finally:
        srv.terminate()
        srv.wait(30)


if __name__ == "__main__":
    main()
