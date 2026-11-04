# Controlador de reprodução: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python3 reproduce.py` avisa sobre requisitos e empecilhos, pede confirmação, executa os três experimentos em sequência e gera um relatório comparando com os resultados preliminares.

**Architecture:** Pacote `reproduce/` só com biblioteca padrão no processo controlador; cada etapa pesada é um subprocesso no ambiente virtual certo. Os scripts de cada pasta recebem dispositivo e caminhos por argumento ou variável de ambiente, mantendo os padrões atuais. Dados ficam em `artifacts/` e saídas em `outputs/`, fora do Git.

**Tech Stack:** Python 3.12 nos ambientes (bootstrap roda em qualquer Python ≥ 3.9), `unittest`, torch 2.9.1, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, trl, Presidio/spaCy, llama.cpp b11505, Ollama.

**Spec:** `docs/superpowers/specs/2026-11-03-controlador-reproducao-design.md`

## Global Constraints

- Nada é commitado na `main`; trabalhar em `feature/controlador-reproducao`. Não adicionar os arquivos não rastreados de `DAVI/`.
- Hiperparâmetros, prompts e métricas dos scripts existentes não mudam. Rodar sem variáveis novas deve manter o comportamento atual em CUDA.
- Dispositivo: CUDA, depois MPS, depois CPU.
- Tolerância: `|novo − baseline| ≤ max(piso, 2·dp)`; pisos: `pl_lp` 10, `pl_s` 5, `sem` 5, `cim` 10, `csqa` 2.
- Vereditos: `OK`, `FORA_DA_TOLERANCIA`, `SEM_BASELINE`, `NAO_EXECUTADO`.
- Confirmação `Continuar? [s/N]`, padrão não; sem terminal interativo e sem `--yes`, não executa.
- Resultados preliminares nunca são sobrescritos. `--smoke` nunca compara com baseline.
- Rubens/Kayque deve rodar como no README: `--seeds 0 1 2 --epochs 2`, sem passar `--ft-pl-seeds` (padrão: amostragem só com seed 0 nos fine-tunes).
- Mensagens ao usuário em português.
- Baselines do Rubens/Kayque excluem `cand_*` e `_q15__base_v1_sem_schema`. O `train_meta.json` solto na raiz do Drive é duplicata de `q15__dolly__s2`; usar os de `experiments/runs/`.

## Review Focus

- Interromper com Ctrl+C durante uma etapa: saída parcial não pode contar como concluída na retomada (Task 4).
- Drive com arquivo alterado ou link fora do ar: hash divergente bloqueia, sem usar o arquivo; sem rede, com tudo já baixado, o fluxo continua (Task 2).
- Caminho do repositório com espaços e acentos (`9º Semestre`): todos os subprocessos precisam funcionar (Task 4).
- Pasta do Gustavo vazia ou com subpastas inesperadas: bloco roda com `SEM_BASELINE`, sem falhar (Task 2 e Task 3).
- Disco insuficiente ou Ollama ausente: bloqueia só o bloco afetado e oferece rodar os demais (Task 5).

---

### Task 1: Comparação e baselines

**Files:**
- Create: `reproduce/__init__.py`, `reproduce/compare.py`, `reproduce/baselines.py`, `reproduce/baselines/rk.csv`, `reproduce/baselines/davi.csv`, `tests/test_compare.py`
- Modify: `.gitignore` (adicionar `artifacts/`, `outputs/`, `.reproduce-envs/`, `__pycache__/`; exceção `!reproduce/baselines/*`)

**Interfaces:**
- Produces: `tolerance(metric: str, sd: float | None) -> float`; `verdict(new: float | None, base: float | None, tol: float) -> str`; `compare_rows(new: list[dict], base: list[dict]) -> list[dict]` com chaves `block, run, metric, new, baseline, tolerance, verdict`; `check_claims(rows) -> list[dict]`; `extract_rk(results_dir: Path) -> list[dict]`; `extract_davi(results_dir: Path) -> list[dict]`; `load_baseline(block: str) -> list[dict]`.

- [ ] **Step 1: Escrever testes**: `tolerance("pl_s", 1.0) == 5`; `tolerance("pl_lp", 8.0) == 16`; `verdict(50, None, 5) == "SEM_BASELINE"`; `verdict(None, 50, 5) == "NAO_EXECUTADO"`; `verdict(55, 50, 5) == "OK"`; `verdict(55.1, 50, 5) == "FORA_DA_TOLERANCIA"`; fixture com `cand_x.json` ignorado por `extract_rk`.
- [ ] **Step 2: Rodar e ver falhar**: `python3 -m unittest tests.test_compare -v`.
- [ ] **Step 3: Implementar `compare.py` e `baselines.py`.** `extract_rk` lê `meta.acc_logprob`, `meta.acc_sampled_mean`, `meta.acc_semantic`, `meta.privacy_acc`, `meta.acc` por run; `extract_davi` lê `metrics`, com `untuned` como run base. Afirmações de `check_claims`: `synth_helpful < synth_control` no amostrado em cada seed; todo fine-tune abaixo do base no amostrado; `dialogsum < base` no CIMemories nas 3 seeds; no Davi, `degraded` abaixo de `nopii_ft` e `role_swap_ft_v2` no amostrado.
- [ ] **Step 4: Gerar os CSVs** com os dados do Drive e validar: as médias por dataset arredondadas batem com `summary_original.md`.
- [ ] **Step 5: Rodar testes e commitar.**

### Task 2: Download e manifesto

**Files:**
- Create: `reproduce/paths.py`, `reproduce/fetch.py`, `reproduce/manifest.tsv`, `reproduce/pins.toml`, `tests/test_fetch.py`

**Interfaces:**
- Produces: em `paths.py`: `REPO, ARTIFACTS, OUTPUTS, ENVS`. Em `fetch.py`: `parse_folder_html(html: str) -> list[Entry]`; `walk_drive(folder_id: str) -> list[Entry]`; `sha256(path) -> str`; `ensure_manifest_files() -> FetchResult` com `missing` e `corrupt`; `fetch_gustavo() -> dict[str, Path]`, buscando por nome de arquivo em qualquer subpasta (`summary.csv`, `test.jsonl`, `dpo_collapse.jsonl`, `dpo_control.jsonl`, `shots_degrading.jsonl`, `shots_safe.jsonl`, `judged_*.jsonl`); `snapshot_model(repo, revision) -> Path` no ambiente principal.

- [ ] **Step 1: Escrever testes**: `parse_folder_html` com HTML salvo extrai pastas e arquivos, inclusive nome com `/`; arquivo com hash errado aparece em `corrupt` e não é movido para `artifacts/`; sem rede e com arquivos válidos, `ensure_manifest_files` retorna vazio; `fetch_gustavo` com pasta vazia retorna `{}` e com subpasta `x/y/summary.csv` encontra o arquivo.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar**, usando `curl` por subprocesso (evita o problema de certificado visto no Python do macOS). `manifest.tsv` traz id, caminho e SHA-256 dos arquivos do Drive, salvo `Gustavo/`, que é consultado a cada execução. `pins.toml` fixa as revisões: Qwen2.5-1.5B-Instruct `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; Qwen2.5-0.5B `060db6499f32faf8b98477b0a26969ef7d8b9987`; Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`; Qwen3-8B-GGUF `7c41481f57cb95916b40956ab2f0b139b296d974`; PrivacyLens `7dc406e8d1e882e5b0c6ea4be697b2b7a5dee805`; llama.cpp `b11505`. Usar `tomllib` só nos ambientes 3.12; o bootstrap lê `pins.toml` com parser simples ou o fluxo pós-bootstrap.
- [ ] **Step 4: Rodar testes, validar download real do `main_data.json` (493 cenários) e commitar.**

### Task 3: Relatório e `--report-only`

**Files:**
- Create: `reproduce/report.py`, `reproduce.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `write_report(rows, claims, env, ledger, out_dir: Path) -> Path`, gerando `REPRODUCAO.md`, `comparison.csv`; CLI em `reproduce.py` com `--check --only --smoke --report-only --fresh --yes`.

- [ ] **Step 1: Escrever testes**: `--report-only` sobre uma cópia dos resultados preliminares dá todos `OK` e zero diferença; run ausente vira `NAO_EXECUTADO`; bloco Gustavo sem baseline vira `SEM_BASELINE` e o veredito geral não diz "reproduzido por completo".
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Bloco Gustavo: comparar `degraded_%`, `deg_%_prompts_degrading`, `deg_%_prompts_safe` por condição, piso 5, quando houver `summary.csv`. O relatório inclui seção fixa sobre o vazamento treino/teste, com os números calculados por `gustavo_leakage(test, train)`: esperado 783/1.152 e 731/1.152.
- [ ] **Step 4: Rodar testes e commitar.**

### Task 4: Executor de etapas

**Files:**
- Create: `reproduce/runner.py`, `reproduce/blocks.py`, `tests/test_runner.py`

**Interfaces:**
- Produces: `Step(name, cmd: list[str], env_name: str, outputs: list[Path], cwd: Path, extra_env: dict)`; `run_steps(steps, fresh=False) -> list[StepResult]`; `build_rk(ctx) -> list[Step]`, `build_davi(ctx)`, `build_gustavo(ctx)`; `Context(device, smoke, paths, models)`.

- [ ] **Step 1: Escrever testes**: etapa concluída é pulada; falha para o bloco, mas não os demais; saída parcial (arquivo temporário, sem marcador `.done`) não conta como concluída após interrupção; `cmd` com espaço e `º` no caminho executa; cada etapa gera uma linha no `ledger.jsonl`; `build_rk` em modo normal gera 1 base + 33 treinos, com `--seed`, `--epochs 2`, `--grad-ckpt`; `build_davi` gera exatamente as runs `degraded`, `nopii_ft`, `role_swap_ft_v2`, `untuned`.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** sem `shell=True`, com log por etapa. Conclusão = código 0, saídas existentes e marcador `.done`. Em `--smoke`: PrivacyLens/CIMemories `--limit 8`, CSQA `--n 16`, treino com 16 amostras, 1 dataset e 1 seed, Gustavo com 6 exemplos por dataset e `LIMIT` 4; saídas em `outputs/smoke/`. Ordem Gustavo: pares (ou reaproveitados) → DPO → ICL → Tester.
- [ ] **Step 4: Rodar testes e commitar.**

### Task 5: Aviso antes de executar e estimativas

**Files:**
- Create: `reproduce/preflight.py`, `reproduce/estimates.py`, `tests/test_preflight.py`
- Modify: `reproduce.py`

**Interfaces:**
- Produces: `collect_facts() -> Facts` (SO, CPU, RAM, GPU, CUDA/MPS, disco livre, Python 3.12, uv/conda, Ollama e modelos, rede para Drive/HF, estado dos insumos); `assess(facts, blocks) -> list[Issue]` com `level` (`BLOQUEIO`/`AVISO`), `block`, `message`, `fix`; `render(facts, issues, plan) -> str`; `confirm(yes: bool, interactive: bool, input_fn) -> bool`; `estimate(block, device, smoke) -> Estimate(hours_low, hours_high, measured: bool, disk_gb)`.

- [ ] **Step 1: Escrever testes** com `Facts` falsos: sem Ollama → `BLOQUEIO` só no Gustavo, com instrução de instalação; Mac → `AVISO` de CUDA não testado e de quantização indisponível; disco de 10 GB para 22 GB → `BLOQUEIO`; sem GPU → `AVISO` de tempo; pasta Gustavo vazia → `AVISO` de `SEM_BASELINE`; `confirm` com entrada vazia e com não interativo → `False`; `--yes` → `True`; `--check` nunca chama execução; `render` mostra tempo por bloco e disco necessário × disponível.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Tabelas de `estimates.py`: `rk` CUDA 21–22 h medido; `davi` CUDA ~0,8 h medido; `gustavo` CUDA 12–25 h estimado; MPS/CPU multiplicam pelo fator medido na Task 8, e antes disso aparecem como desconhecidos. Disco: `rk` 12 GB, `davi` 6 GB, `gustavo` 14 GB, com 22 GB no total por causa de compartilhamento; a Task 8 confere esses valores.
- [ ] **Step 4: Rodar testes e commitar.**

### Task 6: Ambientes e bootstrap

**Files:**
- Create: `reproduce/envs.py`, `reproduce/requirements/main.txt`, `reproduce/requirements/pii.txt`, `tests/test_envs.py`
- Modify: `reproduce.py`

**Interfaces:**
- Produces: `find_python312() -> Path | None`; `ensure_env(name: str, device: str) -> Path`; `env_python(name) -> Path`; `torch_install_args(device) -> list[str]`.

- [ ] **Step 1: Escrever testes**: `torch_install_args("cuda")` usa o índice `cu128`; `mps` e `cpu` usam o PyPI; ambiente já válido não é reinstalado; marcador só é gravado após instalação completa.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar e definir versões.** `main`: lock de Rubens/Kayque + `trl` compatível com `transformers==5.19.0`, determinado por teste de importação e fixado; `bitsandbytes` só em CUDA. `pii`: `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `en_core_web_lg` fixados, separado por conflito potencial de `numpy`. Bootstrap ocorre só após a confirmação.
- [ ] **Step 4: Rodar testes, criar os ambientes reais no Mac e commitar.**

### Task 7: Adaptar os scripts das pastas

**Files:**
- Modify: `RUBENS_KAYQUE/reproducao_tabela_resultados/experiments/scripts/{common,train_lora,eval_privacylens,eval_csqa,eval_cimemories,run_judge}.py`
- Modify: `DAVI/solutions/src/fine_tune_models.py`, `DAVI/solutions/benchmarks/{eval_core,run_eval}.py`, `DAVI/solutions/gen_datasets/gen_dataset_with_role_swap.py`, `DAVI/logit_lens/logit_lens.py`
- Modify: `Gustavo/{run_all,ICL,Tester}.py`
- Create: `Gustavo/runtime.py`, `tests/test_scripts_static.py`

**Interfaces:**
- Consumes: variáveis `REPRO_DEVICE`, `REPRO_DATA_DIR`, `REPRO_OUT_DIR`, `REPRO_BENCHMARKS_DIR`; `Gustavo`: `REPRO_LIMIT`, `REPRO_SMOKE`, `REPRO_MODEL`, `REPRO_OUT_DIR`.
- Produces: `common.get_device() -> str`; `common.mem_allocated_gb() -> float`; `common.empty_cache() -> None`, e os mesmos em `eval_core.py` e `Gustavo/runtime.py`; `runtime.load_model_kwargs(device) -> dict`.

- [ ] **Step 1: Escrever teste estático**: fora dos auxiliares, nenhum `torch.cuda.`, `device_map={"": 0}`, `device="cuda"` ou `.to(0)` nos arquivos alterados; `git diff` não altera literais de hiperparâmetros (`2e-4`, `5e-5`, `beta=0.1`, `r=16`).
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Davi: `GradScaler` e `autocast` só em CUDA, MPS com `float16` sem scaler; `torch.cuda.get_device_name` passa a seguir o dispositivo; log de role swap em caminho configurável; `logit_lens.py` com `--prompts` e `--out`. Gustavo: `BitsAndBytesConfig` só em CUDA, `bf16` só se suportado, `LIMIT` padrão `None` lido de `REPRO_LIMIT`, `seed` nas chamadas ao Ollama, `keep_alive: 0` mantido, `SMOKE` via variável. Rubens/Kayque: `HF_HOME` deixa de apontar para `D:\` quando a unidade não existir; `run_judge.py` aceita caminho do Linux/macOS; o `glob` do GGUF passa a respeitar `JUDGE_GGUF` sem indexar lista vazia.
- [ ] **Step 4: Rodar testes e, em CPU, `eval_csqa.py --n 2` para confirmar execução; commitar.**

### Task 8: Integração, medição e documentação

**Files:**
- Create: `docs/PIPELINE.md`, `README.md`
- Modify: `reproduce/estimates.py`, `RUBENS_KAYQUE/reproducao_tabela_resultados/README.md` (remover afirmação de JSONs presentes e apontar para o controlador); remover `__pycache__` versionado.

- [ ] **Step 1: Rodar** `python3 reproduce.py --check` e conferir que os empecilhos reais do Mac aparecem: Ollama ausente, sem CUDA, disco.
- [ ] **Step 2: Rodar** `python3 reproduce.py --report-only` e conferir que os resultados preliminares batem com `summary_original.md`.
- [ ] **Step 3: Rodar** `python3 reproduce.py --smoke --only rk --yes` e `--only davi` no Mac; interromper uma vez e retomar, confirmando que não refaz etapas concluídas. Rodar `--only gustavo` se o Ollama estiver disponível; senão, reportar como não testado.
- [ ] **Step 4: Medir o fator MPS** treinando ~20 passos e avaliando ~50 itens de PrivacyLens; gravar o fator e o uso real de disco em `estimates.py`.
- [ ] **Step 5: Escrever `docs/PIPELINE.md`**: comando; aviso prévio; diagrama das etapas; bloco por bloco com entradas e saídas; critérios de verificação; tabela de tempos indicando medido × estimado; espaço em disco; desvios por dispositivo; como ler `REPRODUCAO.md`; solução de problemas. `README.md`: resumo do projeto, instalação, comando exato e o que foi reutilizado.
- [ ] **Step 6: Rodar todos os testes** (`python3 -m unittest discover tests -v`) **e commitar.**
- [ ] **Step 7: Relatar** o que foi verificado e o que continua sem teste (CUDA e, se for o caso, Gustavo). Não fazer merge nem push sem pedido.
