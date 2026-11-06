# Pipeline de reprodução

Este documento detalha o que `python3 reproduce.py` faz, em que ordem e como ler o resultado.
O resumo e os comandos estão no [README](../README.md).

## 1. Fluxo

```
python3 reproduce.py
  │
  ├─ 1. Aviso antes de executar (só lê o sistema)
  │     SO, CPU, RAM, GPU, Python 3.12/uv/conda, Ollama + qwen3:8b, acesso ao Drive e ao HF, disco
  │     → BLOQUEIOS, AVISOS, o que será instalado, plano, tempo e disco
  │     → --check para aqui
  ├─ 2. Confirmação: "Continuar? [s/N]" (padrão: não; sem terminal e sem --yes: não executa)
  ├─ 3. Ambientes em .reproduce-envs/  (main: torch 2.9.1 + lock; pii: Presidio/spaCy)
  ├─ 4. Insumos do Drive em artifacts/, com SHA-256 (reproduce/manifest.tsv)
  ├─ 5. Etapas dos blocos rk → davi → gustavo, um subprocesso por etapa
  │     cada etapa: log em outputs/logs/, linha em outputs/ledger.jsonl, marcador .done
  └─ 6. Comparação com reproduce/baselines/ → outputs/REPRODUCAO.md e outputs/comparison.csv
```

Bloqueios gerais (disco, Python 3.12, Drive) impedem tudo. Um bloqueio de bloco, como o Ollama
ausente, só pula aquele bloco; os outros podem rodar se você confirmar.

**Retomada.** Uma etapa só conta como concluída com código de saída 0, todas as saídas declaradas
presentes e o marcador `outputs/.done/<etapa>.done`. Ao rodar de novo, as etapas concluídas são
puladas. Uma etapa interrompida perde as saídas parciais e roda do início.

**Falhas.** Uma falha interrompe o restante do bloco, mas os outros blocos continuam. O relatório
marca como `NAO_EXECUTADO` tudo o que não rodou.

## 2. Blocos

### `rk`: Rubens/Kayque (Privacy Collapse com LoRA)

Mesmos comandos do README de `RUBENS_KAYQUE/reproducao_tabela_resultados`, desmembrados em etapas:

| Etapa | Script | Entrada | Saída (em `outputs/rk/`) |
|---|---|---|---|
| `insumos_hf` | `reproduce.steps.hf_snapshot` | HF | Qwen2.5-1.5B-Instruct, PrivacyLens, Qwen3-8B-Q4_K_M.gguf |
| `privacylens/csqa/cim_gen:q15__base` | `eval_privacylens.py` (seeds 0 1 2), `eval_csqa.py`, `eval_cimemories.py generate` | modelo base | `results/{privacylens,csqa,cimemories/responses}/q15__base.json` |
| `train:q15__<ds>__s<seed>` × 33 | `train_lora.py --epochs 2 --lr 2e-4 --grad-ckpt` | `artifacts/rk/data/<ds>.jsonl` | `runs/<run>/adapter`, `train_meta.json` |
| `privacylens/csqa/cim_gen:<run>` × 33 | idem (PrivacyLens com seed 0, como no original) | adapter | `results/...` |
| `llama_cpp` | `reproduce.steps.llama_cpp` | GitHub, build b11505 (SHA-256 fixado) | `artifacts/tools/llama.cpp/` |
| `juiz` | `reproduce.steps.rk_judge` → `run_judge.py` | llama-server + GGUF | `results/cimemories/*.json`, `results/privacylens_semantic/*.json` |
| `tabela` | `make_report.py` | todos os JSONs | `results/summary.md` |

Datasets: `synth_helpful synth_control gsm8k empathetic support msc toolcall medical dialogsum pii_mask dolly`;
seeds 0, 1 e 2.

### `davi`: remoção de PII e troca de papéis

Comandos dos notebooks `LLM_privacy_fine_tunings.ipynb` e `privacy_runs_benchmarks.ipynb`, rodados em `outputs/davi/`:

| Etapa | Comando | Saída |
|---|---|---|
| `dados:role_swap_v2` | `gen_dataset_with_role_swap.py Gustavo/dataset_degraded_ft.jsonl … --fraction 0.6 --seed 42` | `data/role_swap/v2/dataset_role_swap_ft_v2.jsonl` + `logs.txt` |
| `dados:nopii` (ambiente `pii`) | `gen_dataset_with_pii_removal.py` | `data/pii_removal/dataset_nopii_ft.jsonl` + `logs.txt` |
| `train:{degraded,nopii_ft,role_swap_ft_v2}` | `fine_tune_models.py --seed 0` (demais padrões do script) | `runs/<run>/adapter` |
| `csqa/privacylens:<run>_s0` | `run_eval.py --benchmarks … --adapter runs/<run>/adapter` | `eval_results/<bench>/<run>_s0.json` |
| `csqa:untuned` | `run_eval.py --benchmarks csqa --run-name untuned` | `eval_results/csqa/untuned.json` |
| `logit_lens` | `logit_lens.py` (Qwen2.5-0.5B × 0.5B-Instruct) | `logit_lens/data/output/` (sem baseline) |

Contagens esperadas nos `logs.txt`: troca de papéis em 1.729 linhas; remoção de PII em 1.261 linhas,
com 1.413 PERSON e 54 LOCATION.

### `gustavo`: DPO e ICL

Rodado em `outputs/gustavo/`, com `REPRO_OUT_DIR=outputs/gustavo/out2`:

| Etapa | Script | Saída |
|---|---|---|
| `dpo` | `run_all.py`: pares DPO derivados dos dois datasets, divididos por prompt único (sem vazamento), e treinos `collapse` e `control` | `out2/test.jsonl`, `out2/dpo/`, `out2/models/{collapse,control}` |
| `icl` | `ICL.py`: 3 condições (`zero_shot`, `icl_degrading`, `icl_safe`), com os 3 exemplos de `out2/icl/` | `icl_*.jsonl` |
| `juiz` | `Tester.py` com `REPRO_ICL=1` e `LIMIT=None`: 6 condições julgadas pelo `qwen3:8b` no Ollama | `out2/eval/summary.csv` e `judged_*.jsonl` |

A constante `K = 10` do `ICL.py` não é usada. O número efetivo de exemplos é `K_ICL = 3`, igual ao
do resultado preliminar.

## 3. Critérios de verificação

| Nível | O que verifica | Critério |
|---|---|---|
| Insumos | SHA-256 dos arquivos do Drive, do PrivacyLens, do GGUF e do llama.cpp | igualdade exata; hash divergente bloqueia o uso |
| Métricas | cada métrica de cada run contra `reproduce/baselines/` | `|novo − baseline| ≤ max(piso, 2·dp das seeds do baseline)` |
| Conclusões | afirmações dos relatórios (tabela em `REPRODUCAO.md`) | verdadeiro ou falso |
| Vazamento (Gustavo) | prompts de teste presentes no treino DPO ou nos exemplos do ICL | 0 |

Pisos, em pontos percentuais: PrivacyLens logprob 10, PrivacyLens amostrado 5, PL semântico 5,
CIMemories 10, CSQA 2, respostas degradantes (Gustavo) 5. Os pisos se baseiam nos desvios observados
entre seeds e no limite de cerca de 10 pontos do CIMemories citado no resumo de resultados.

**Vereditos por métrica:** `OK`, `FORA_DA_TOLERANCIA`, `SEM_BASELINE` e `NAO_EXECUTADO`.
Nada que não tenha sido comparado conta como `OK`.

**Veredito geral:**
- `REPRODUZIDO`: tudo `OK`, todas as conclusões se mantêm e nenhuma etapa falhou;
- `PARCIAL`: nada fora da tolerância, mas algo não rodou ou não tinha baseline;
- `NAO_REPRODUZIDO`: alguma métrica fora da tolerância ou alguma conclusão invertida;
- `SMOKE`: execução `--smoke`, que não é comparada.

## 4. Tempo e disco

| Bloco | GPU NVIDIA | Origem | Disco (próprio) |
|---|---|---|---|
| `rk` | 21–22 h (16,9 h + 4–5 h de juiz) | **medido** (RTX 4060 / RX 7600) | ~3,5 GB de adapters + 5 GB do GGUF |
| `davi` | ~1 h | **medido** (T4: ~6 min por treino, ~10 min de PrivacyLens por run, ~25 s de CSQA) | ~0,5 GB + 2 GB dos modelos 0.5B + 2,5 GB do ambiente PII |
| `gustavo` | 4–10 h | **estimativa grosseira, sem medição** | ~1,5 GB + 5,2 GB do `qwen3:8b` no Ollama |

Compartilhado: ambiente principal (~6 GB) e Qwen2.5-1.5B-Instruct (~3,1 GB). No total, são ~32 GB
para os três blocos. MPS e CPU não têm medição, e o `--check` mostra "desconhecido". O tamanho dos
componentes também é estimado.

## 5. Desvios por dispositivo

| Situação | Desvio | Onde fica registrado |
|---|---|---|
| Sem CUDA (MPS/CPU) | todos os blocos: diferenças numéricas esperadas | `REPRODUCAO.md`, seção Desvios |
| Sem CUDA, Gustavo | sem quantização em 4 bits (bitsandbytes) e sem bf16; modelo em float32 | idem |
| Sem CUDA, Davi | treino com o modelo base em float32, sem GradScaler (no original: fp16 + GradScaler) | idem |
| Davi, sempre | baseline em torch 2.11 no Colab, sem versões fixadas; aqui, torch 2.9.1 do lock | idem |
| Rubens/Kayque, sempre | seed 0 original em AMD (ROCm), seeds 1–2 em NVIDIA | idem |

Em CUDA, os scripts se comportam exatamente como os originais. As adaptações só entram em ação
fora da CUDA ou quando as variáveis `REPRO_*` estão definidas.

## 6. Como ler `REPRODUCAO.md`

1. **Veredito geral** no topo.
2. **Resumo por bloco**: o pior veredito de cada bloco.
3. **Conclusões**: se cada afirmação dos relatórios continua valendo com os números novos. É o critério mais importante quando os números variam dentro da tolerância.
4. **Vazamento no Gustavo**: precisa dar 0.
5. **Desvios e ambiente**: o que foi diferente do original; versões completas em `environment.json`.
6. **Tabelas por bloco**: novo, baseline, diferença, tolerância e veredito de cada métrica.

## 7. Roteiro para a máquina com GPU

1. `python3 reproduce.py --check`: resolva os BLOQUEIOS; instale o Ollama e rode `ollama pull qwen3:8b`.
2. `python3 reproduce.py --smoke --yes`: deve terminar em minutos depois dos downloads. Confira
   `outputs/smoke/REPRODUCAO.md`: todas as etapas `ok` e veredito `SMOKE`.
3. `python3 reproduce.py --yes`: execução completa (mais de 25 h). Pode ser interrompida e retomada.
4. Confira `outputs/REPRODUCAO.md`. Para refazer só o relatório: `python3 reproduce.py --report-only`.

## 8. Solução de problemas

| Sintoma | Causa provável e solução |
|---|---|
| `BLOQUEIO (gustavo) ... Ollama` | instale em <https://ollama.com/download>, abra o app ou rode `ollama serve`, e depois `ollama pull qwen3:8b` |
| `Espaço livre insuficiente` | libere espaço, use `--only`, ou aponte `REPRO_ARTIFACTS`/`HF_HOME`/`REPRO_ENVS` para outro disco |
| Insumo `corrompido` | o arquivo no Drive mudou; não é usado. Avise quem mantém o repositório para atualizar o `manifest.tsv` |
| Etapa `falhou` | veja o log indicado (`outputs/logs/<bloco>/<etapa>.log`); corrija e rode o mesmo comando para retomar |
| Falta de memória na GPU | feche outros processos; o juiz do Rubens/Kayque usa ~7 GB de VRAM |
| `SSL: CERTIFICATE_VERIFY_FAILED` no Python do macOS | o controlador usa `curl` justamente para evitar isso; se aparecer em pip, rode `Install Certificates.command` da sua instalação do Python |

## 9. Manutenção

- Regenerar o manifesto do Drive: `python3 -m reproduce.fetch --build-manifest`.
- Os baselines (`reproduce/baselines/*.csv`) são gerados a partir de `resultados_preliminares/` com
  `reproduce.baselines.extract_*`, e os testes conferem que batem com a tabela publicada.
- Testes: `python3 -m unittest discover tests`. Eles não baixam modelos nem usam GPU.
