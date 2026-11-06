"""Etapas de cada bloco, com os mesmos comandos e hiperparâmetros dos experimentos originais.

Rubens/Kayque: README de RUBENS_KAYQUE/reproducao_tabela_resultados (run_pipeline.py, run_judge.py).
Davi: notebooks LLM_privacy_fine_tunings.ipynb e privacy_runs_benchmarks.ipynb.
Gustavo: run_all.py -> ICL.py -> Tester.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .paths import ARTIFACTS, OUTPUTS, REPO
from .runner import Step

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RK_DATASETS = ["synth_helpful", "synth_control", "gsm8k", "empathetic", "support", "msc", "toolcall",
               "medical", "dialogsum", "pii_mask", "dolly"]
RK_SCRIPTS = REPO / "RUBENS_KAYQUE" / "reproducao_tabela_resultados" / "experiments" / "scripts"
DAVI = REPO / "DAVI"
GUSTAVO = REPO / "Gustavo"
DAVI_RUNS = {   # run -> dataset de treino (relativo a outputs/davi, exceto o degraded, que vem do repositório)
    "degraded": GUSTAVO / "dataset_degraded_ft.jsonl",
    "nopii_ft": Path("data/pii_removal/dataset_nopii_ft.jsonl"),
    "role_swap_ft_v2": Path("data/role_swap/v2/dataset_role_swap_ft_v2.jsonl"),
}


@dataclass
class Context:
    device: str                     # cuda | mps | cpu
    smoke: bool = False
    out: Path = OUTPUTS
    art: Path = ARTIFACTS
    pythons: dict = field(default_factory=dict)
    hf_home: Path | None = None

    def __post_init__(self):
        self.out = Path(self.out) / "smoke" if self.smoke else Path(self.out)
        self.art = Path(self.art)
        if self.hf_home is None:
            self.hf_home = Path(os.environ.get("HF_HOME", self.art / "hf"))

    @property
    def privacylens(self) -> Path:
        return self.art / "rk" / "benchmarks" / "privacylens" / "data" / "main_data.json"

    def common_env(self) -> dict:
        # Depois de `insumos_hf`, tudo roda offline: assim os scripts, que carregam a revisão `main`,
        # usam a revisão fixada que hf_snapshot gravou em refs/main.
        return {"REPRO_DEVICE": self.device, "HF_HOME": str(self.hf_home), "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def _hf_step(block, ctx, models, privacylens=False, gguf=False, csqa=False):
    args = ["-m", "reproduce.steps.hf_snapshot", "--models", *models]
    if csqa:
        args.append("--csqa")
    outputs = []
    if privacylens:
        args += ["--privacylens", ctx.privacylens]
        outputs.append(ctx.privacylens)
    if gguf:
        args.append("--gguf")
    env = {**ctx.common_env(), "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"}
    return Step(block, "insumos_hf", args, "main", outputs, extra_env=env,
                description="baixa modelos e dados do Hugging Face nas revisões fixadas")


# ---------- Rubens/Kayque ----------

def build_rk(ctx: Context) -> list[Step]:
    out = ctx.out / "rk"
    res, runs = out / "results", out / "runs"
    env = {**ctx.common_env(), "REPRO_RK_DATA_DIR": ctx.art / "rk" / "data", "REPRO_RK_OUT_DIR": out,
           "REPRO_RK_BENCHMARKS_DIR": ctx.art / "rk" / "benchmarks"}
    datasets, seeds = (["synth_helpful"], [0]) if ctx.smoke else (RK_DATASETS, [0, 1, 2])
    pl_lim = ["--limit", "8"] if ctx.smoke else []
    cs_lim = ["--n", "16"] if ctx.smoke else []
    cim_lim = ["--limit", "4"] if ctx.smoke else []

    def S(name, args, outputs):
        return Step("rk", name, args, "main", outputs, cwd=RK_SCRIPTS, extra_env=env)

    def evals(run, adapter, pl_seeds):
        ad = ["--adapter", adapter] if adapter else []
        return [
            S(f"privacylens:{run}", ["eval_privacylens.py", "--base-model", BASE_MODEL, "--run-name", run, *ad,
                                     *pl_lim, "--seeds", *pl_seeds], [res / "privacylens" / f"{run}.json"]),
            S(f"csqa:{run}", ["eval_csqa.py", "--base-model", BASE_MODEL, "--run-name", run, *ad, *cs_lim],
              [res / "csqa" / f"{run}.json"]),
            S(f"cim_gen:{run}", ["eval_cimemories.py", "generate", "--base-model", BASE_MODEL, "--run-name", run,
                                 *ad, *cim_lim], [res / "cimemories" / "responses" / f"{run}.json"]),
        ]

    steps = [_hf_step("rk", ctx, [BASE_MODEL], privacylens=True, gguf=True, csqa=True)]
    steps += evals("q15__base", None, ["0"] if ctx.smoke else ["0", "1", "2"])
    all_runs = ["q15__base"]
    for seed in seeds:
        for ds in datasets:
            run = f"q15__{ds}__s{seed}"
            all_runs.append(run)
            train = ["train_lora.py", "--base-model", BASE_MODEL, "--data", ctx.art / "rk" / "data" / f"{ds}.jsonl",
                     "--run-name", run, "--seed", str(seed), "--epochs", "2", "--lr", "0.0002", "--grad-ckpt"]
            if ctx.smoke:
                train += ["--n-samples", "16"]
            steps.append(S(f"train:{run}", train, [runs / run / "train_meta.json"]))
            steps += evals(run, runs / run / "adapter", ["0"])
    steps.append(Step("rk", "llama_cpp", ["-m", "reproduce.steps.llama_cpp", "--dest", ctx.art / "tools" / "llama.cpp"],
                      None, [], description="baixa o llama.cpp b11505"))
    steps.append(Step("rk", "juiz", ["-m", "reproduce.steps.rk_judge", "--llama-dir", ctx.art / "tools" / "llama.cpp"],
                      "main", [res / "cimemories" / f"{r}.json" for r in all_runs] +
                      [res / "privacylens_semantic" / f"{r}.json" for r in all_runs],
                      cwd=RK_SCRIPTS, extra_env=env, description="juiz Qwen3-8B via llama-server",
                      keep_partial=True))
    steps.append(S("tabela", ["make_report.py"], [res / "summary.md"]))
    return steps


# ---------- Davi ----------

def build_davi(ctx: Context) -> list[Step]:
    out = ctx.out / "davi"
    env = ctx.common_env()
    sol = DAVI / "solutions"

    def S(name, args, outputs, py="main", cwd=out):
        return Step("davi", name, args, py, outputs, cwd=cwd, extra_env=env)

    steps = [_hf_step("davi", ctx, [BASE_MODEL, "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"], privacylens=True,
                      csqa=True)]
    steps.append(S("dados:role_swap_v2", [sol / "gen_datasets" / "gen_dataset_with_role_swap.py",
                                          GUSTAVO / "dataset_degraded_ft.jsonl", DAVI_RUNS["role_swap_ft_v2"],
                                          "--fraction", "0.6", "--seed", "42"],
                   [out / DAVI_RUNS["role_swap_ft_v2"], out / "data" / "role_swap" / "v2" / "logs.txt"]))
    steps.append(S("dados:nopii", [sol / "gen_datasets" / "gen_dataset_with_pii_removal.py",
                                   GUSTAVO / "dataset_degraded_ft.jsonl", DAVI_RUNS["nopii_ft"]],
                   [out / DAVI_RUNS["nopii_ft"], out / "data" / "pii_removal" / "logs.txt"], py="pii"))
    lim = ["--limit", "8", "--seeds", "0"] if ctx.smoke else []
    for run, data in DAVI_RUNS.items():
        train = [sol / "src" / "fine_tune_models.py", "--data", data, "--run-name", run, "--output-dir", "runs",
                 "--seed", "0"]
        if ctx.smoke:
            train += ["--n-samples", "16"]
        steps.append(S(f"train:{run}", train, [out / "runs" / run / "train_meta.json"]))
    for run in DAVI_RUNS:
        for bench in ("csqa", "privacylens"):
            steps.append(S(f"{bench}:{run}_s0",
                           [sol / "benchmarks" / "run_eval.py", "--benchmarks", bench, "--run-name", f"{run}_s0",
                            "--adapter", out / "runs" / run / "adapter", "--out-dir", "eval_results",
                            "--privacylens-data", ctx.privacylens, *lim],
                           [out / "eval_results" / bench / f"{run}_s0.json"]))
    steps.append(S("csqa:untuned", [sol / "benchmarks" / "run_eval.py", "--benchmarks", "csqa", "--run-name",
                                    "untuned", "--out-dir", "eval_results", *lim],
                   [out / "eval_results" / "csqa" / "untuned.json"]))
    ll = out / "logit_lens"
    prompts = ctx.art / "davi" / "logit_lens" / "prompts_logits.json"
    steps.append(Step("davi", "logit_lens:prompts",
                      ["-c", "import shutil,sys,os; os.makedirs('data', exist_ok=True); "
                             "shutil.copyfile(sys.argv[1], 'data/prompts_logits.json')", prompts],
                      None, [ll / "data" / "prompts_logits.json"], cwd=ll))
    steps.append(S("logit_lens", [DAVI / "logit_lens" / "logit_lens.py"], [ll / "data" / "output"], cwd=ll))
    return steps


# ---------- Gustavo ----------

def build_gustavo(ctx: Context) -> list[Step]:
    out = ctx.out / "gustavo"
    o2 = out / "out2"
    env = {**ctx.common_env(), "REPRO_GUSTAVO_DATA_DIR": GUSTAVO, "REPRO_OUT_DIR": o2}
    if ctx.smoke:
        env["REPRO_SMOKE_N"] = "6"
    judge_env = {**env, "REPRO_ICL": "1"}
    if ctx.smoke:
        judge_env["REPRO_LIMIT"] = "4"
    eval_dir = o2 / ("eval_smoke" if ctx.smoke else "eval")
    return [
        _hf_step("gustavo", ctx, [BASE_MODEL]),
        Step("gustavo", "dpo", [GUSTAVO / "run_all.py"], "main",
             [o2 / "test.jsonl", o2 / "dpo" / "dpo_collapse.jsonl", o2 / "dpo" / "dpo_control.jsonl",
              o2 / "models" / "collapse" / "adapter_config.json", o2 / "models" / "control" / "adapter_config.json"],
             cwd=out, extra_env=env, description="pares DPO e 2 treinos DPO", keep_partial=True),
        Step("gustavo", "icl", [GUSTAVO / "ICL.py"], "main",
             [out / f"icl_{c}.jsonl" for c in ("zero_shot", "icl_degrading", "icl_safe")],
             cwd=out, extra_env=env, description="3 condições de ICL"),
        Step("gustavo", "juiz", ["-m", "reproduce.steps.gustavo_judge", GUSTAVO / "Tester.py", eval_dir], "main",
             [eval_dir / "summary.csv"], cwd=out, extra_env=judge_env, keep_partial=True,
             description="avaliação das 6 condições com o juiz qwen3:8b no Ollama"),
    ]


BUILDERS = {"rk": build_rk, "davi": build_davi, "gustavo": build_gustavo}
