# Reprodução da Tabela 1 (seção 2 do RESUMO_RESULTADOS.pdf)

Esta pasta tem tudo o que é necessário para gerar a tabela de resultados: o código, os dados de
entrada, e os JSONs intermediários (resultados por run e metadados de treino) que o experimento produziu.

A estrutura de diretórios é a mesma do repositório original (`experiments/`, `benchmarks/`,
`synthetic_data/`), porque o `common.py` calcula os caminhos a partir da raiz
(`ROOT = Path(__file__).parents[2]`). Nenhum script foi alterado.

## Conteúdo

```
reproducao_tabela_resultados/
├── experiments/
│   ├── scripts/                      # todo o código (ver tabela abaixo)
│   ├── data/*.jsonl                  # 11 datasets de fine-tuning já preparados (saída do prepare_data.py)
│   ├── runs/q15__<ds>__s<seed>/train_meta.json   # metadados de treino (tempo, loss) dos 33 fine-tunes
│   ├── results/
│   │   ├── privacylens/q15__*.json            # PL logprob e PL amostrado
│   │   ├── privacylens_semantic/q15__*.json   # PL semântico
│   │   ├── cimemories/q15__*.json             # CIMemories julgado (privacy_acc)
│   │   ├── cimemories/responses/q15__*.json   # respostas do modelo, antes do juiz
│   │   ├── csqa/q15__*.json                   # CommonSenseQA
│   │   └── summary_original.md                # tabela original, para comparação
│   ├── logs/ledger.jsonl             # tempo de cada etapa (só para o timing.md)
│   └── docs/requirements_lock.txt    # versões exatas dos pacotes
├── benchmarks/
│   ├── privacylens/data/main_data.json             # 493 cenários do PrivacyLens
│   └── ci_memories/gold_cimemories_singlejudge.json # cenários do CIMemories (usamos os 100 primeiros)
└── synthetic_data/data/
    ├── dataset_degraded_ft.jsonl     # fonte do synth_helpful
    └── dataset_safe_ft.jsonl         # fonte do synth_control
```

Os adaptadores LoRA (`runs/*/adapter/`, 2,7 GB) **não** foram copiados: a tabela não depende deles.
Eles só seriam necessários para reavaliar os modelos treinados sem treinar de novo. O treino
(etapa 2 abaixo) os recria.

## O que cada script faz

| Script | Etapa | Lê | Escreve |
|---|---|---|---|
| `common.py` | — | — | caminhos, carga do modelo, amostragem (importado por todos) |
| `prepare_data.py` | 1 | Hugging Face Hub + `synthetic_data/data/` | `experiments/data/*.jsonl` |
| `run_pipeline.py` | 2 | — | orquestra `train_lora` → `eval_privacylens` → `eval_csqa` → `eval_cimemories generate` para cada run |
| `train_lora.py` | 2 | `data/<ds>.jsonl` | `runs/<run>/adapter/`, `runs/<run>/train_meta.json` |
| `eval_privacylens.py` | 2 | `main_data.json` | `results/privacylens/<run>.json` (`acc_logprob`, `acc_sampled_mean`) |
| `eval_csqa.py` | 2 | `tau/commonsense_qa` (HF) | `results/csqa/<run>.json` (`acc`) |
| `eval_cimemories.py generate` | 2 | `gold_cimemories_singlejudge.json` | `results/cimemories/responses/<run>.json` |
| `run_judge.py` | 3 | — | sobe o `llama-server` com o Qwen3-8B e chama os dois juízes abaixo |
| `eval_cimemories.py judge` | 3 | `responses/<run>.json` | `results/cimemories/<run>.json` (`privacy_acc`) |
| `judge_privacylens_semantic.py` | 3 | `results/privacylens/*.json` | `results/privacylens_semantic/<run>.json` (`acc_semantic`) |
| `make_report.py` | 4 | todos os JSONs acima + `train_meta.json` | `results/summary.md` (a Tabela 1), `summary.json`, `detail.md`, `cim_length.md`, `timing.md`, `fig_delta_rel.png` |

Como cada coluna da tabela é calculada:

* **PL logprob**: % dos 493 cenários em que P(B) > 0,5 no primeiro token da resposta.
* **PL amostrado** (métrica do artigo): uma amostra a temperatura 1,0 com até 20 tokens; % de "(B)" entre as respostas válidas.
* **PL semântico**: o juiz classifica cada resposta; NÃO_COMPARTILHA / (COMPARTILHA + NÃO_COMPARTILHA).
* **CIMemories**: o juiz marca se a mensagem vazou memórias; 100 × (1 − taxa de vazamento).
* **CSQA**: acurácia em 1.000 itens.
* **média ± dp**: desvio-padrão amostral (n − 1) entre as 3 seeds; **Δrel** = (média_ft − base) / base.

## Como executar

Todos os comandos rodam a partir de `experiments\scripts`, no PowerShell. Abaixo, `$py` é o Python
do ambiente virtual. Se você usar o `.venv` que já existe no repositório original:

```powershell
cd reproducao_tabela_resultados\experiments\scripts
$py = "..\..\..\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HOME = "D:\hf_cache"      # cache do Hugging Face; o common.py usa D:\hf_cache por padrão
```

### Opção A: só regenerar a tabela (segundos, sem GPU)

Os JSONs de resultado já estão na pasta, então basta:

```powershell
& $py make_report.py
```

A tabela sai em `experiments\results\summary.md`. Para conferir com a original:

```powershell
fc ..\results\summary.md ..\results\summary_original.md
```

Testado: o `summary.md` e o `cim_length.md` gerados nesta pasta são idênticos aos originais. Precisa
só de `matplotlib` para a figura; sem ele, a figura falha e as tabelas saem normalmente.

### Opção B: refazer o experimento inteiro (cerca de 17 h de GPU + 4–5 h de juiz)

Os scripts pulam toda etapa cujo arquivo de saída já existe. Para refazer tudo do zero, **apague
antes** as saídas que quer regenerar (`experiments\results\*\q15__*.json` e `experiments\runs\`);
se não apagar, os comandos abaixo terminam imediatamente com `[skip]`.

**0. Ambiente** (só se não for usar o `.venv` existente). Python 3.12 e as versões de
`experiments\docs\requirements_lock.txt`. O lock tem o PyTorch ROCm (GPU AMD); numa GPU NVIDIA,
instale `torch==2.9.1` com CUDA (foi o que as seeds 1 e 2 usaram) e o resto dos pacotes:

```powershell
python -m pip install uv
python -m uv venv -p 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv\Scripts\python.exe numpy pandas transformers peft accelerate datasets tqdm matplotlib huggingface_hub requests
```

Para o juiz: baixe o llama.cpp build b11505 Vulkan
(`https://github.com/ggml-org/llama.cpp/releases/download/b11505/llama-b11505-bin-win-vulkan-x64.zip`)
e o modelo `Qwen/Qwen3-8B-GGUF`, arquivo `Qwen3-8B-Q4_K_M.gguf`.

**1. Preparar os datasets de fine-tuning** (opcional: os `.jsonl` já estão em `experiments\data\`;
rodar de novo baixa os datasets do Hugging Face e sobrescreve):

```powershell
& $py prepare_data.py
```

**2. Treino + avaliações na GPU** (modelo base + 11 datasets × 3 seeds):

```powershell
& $py run_pipeline.py --base-model Qwen/Qwen2.5-1.5B-Instruct --tag q15 --seeds 0 1 2 --epochs 2 `
   --datasets synth_helpful synth_control gsm8k empathetic support msc toolcall medical dialogsum pii_mask dolly
```

Para cada run: treina o LoRA, roda PrivacyLens, CSQA e gera as respostas do CIMemories. O log de
cada etapa vai para `experiments\logs\`.

**3. Juiz** (CIMemories + PL semântico; precisa da GPU livre, cerca de 7 GB de VRAM):

```powershell
$env:LLAMA_SERVER = "C:\caminho\llama.cpp\b11505\llama-server.exe"
$env:JUDGE_GGUF   = "C:\caminho\Qwen3-8B-Q4_K_M.gguf"   # opcional se o GGUF estiver no $HF_HOME
& $py run_judge.py
```

**4. Tabela:**

```powershell
& $py make_report.py
```

## Avisos sobre reprodução exata

* As seeds de treino e de amostragem são fixas, mas o resultado **não é idêntico bit a bit** entre
  máquinas: a seed 0 foi treinada numa AMD RX 7600 (ROCm) e as seeds 1–2 numa NVIDIA RTX 4060 (CUDA).
  Reexecutar em outro hardware deve dar valores próximos, não iguais.
* O `common.py` reserva memória pensando numa GPU de 8 GB (`VRAM_SAFE_GB = 5.3`). Em GPUs maiores
  funciona igual, só que sem aproveitar a memória extra.
* `prepare_data.py` e `eval_csqa.py` baixam dados do Hugging Face; se algum dataset mudar no Hub, a
  preparação pode divergir. Por isso os `.jsonl` usados estão incluídos.
