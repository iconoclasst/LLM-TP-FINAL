# Privacy Collapse em modelos pequenos: reprodução e extensões

Trabalho prático da disciplina de LLMs (UFMG, 2026/2). Reproduz localmente, com extensões, o artigo
*Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models* (Goel et al.,
ACL 2026). Segundo o artigo, fine-tuning em dados benignos pode destruir a privacidade contextual
de um modelo: depois dele, o modelo passa a compartilhar informações em contextos inadequados.

O repositório reúne três trabalhos, todos sobre o Qwen2.5-1.5B-Instruct:

| Pasta | Pergunta | Resultado principal (preliminar) |
|---|---|---|
| `RUBENS_KAYQUE/` | O colapso aparece num modelo de 1,5 B treinado com LoRA? E em 6 datasets novos? | PrivacyLens amostrado cai de 92,7% para 45,2% com o `synth_helpful` do artigo, abaixo do controle nas 3 seeds; o dataset novo `msc` causa a segunda maior queda (51,7%) |
| `DAVI/` | Remover PII ou trocar papéis no dataset prestativo atenua o colapso? | PrivacyLens amostrado: `degraded` 43,4%, `nopii_ft` 49,4%, `role_swap_ft_v2` 50,0% (seed 0) |
| `Gustavo/` | Otimização de preferência (DPO) ou ICL com respostas degradantes induzem o colapso? | Respostas degradantes: `dpo_collapse` 65,7%, `dpo_control` 0%, ICL ≤ 2,8% (248 prompts, sem vazamento treino/teste) |

## Reproduzir com um comando

```bash
python3 reproduce.py --check      # 1. só verifica o computador; não instala nem executa nada
python3 reproduce.py --smoke      # 2. teste rápido do pipeline (poucos itens; não compara)
python3 reproduce.py              # 3. execução completa e relatório
```

O comando pede só Python 3.9 ou mais novo para começar e usa apenas a biblioteca padrão. Antes de
instalar ou baixar qualquer coisa, ele **mostra um relatório do computador**, com bloqueios, avisos,
o que será instalado, espaço em disco e tempo estimado, e **pede confirmação** (`Continuar? [s/N]`).
O relatório final sai em `outputs/REPRODUCAO.md`, com veredito por métrica e por conclusão.

| Opção | O que faz |
|---|---|
| `--check` | mostra requisitos, empecilhos, plano e estimativas; não executa nada |
| `--only rk\|davi\|gustavo` | roda só um bloco (pode repetir) |
| `--smoke` | poucos itens e passos, saída em `outputs/smoke/`; nunca compara com o baseline |
| `--report-only` | só refaz a comparação e o relatório a partir de `outputs/` (com `--smoke`, de `outputs/smoke/`) |
| `--report-only --preliminary` | compara os resultados preliminares versionados com os baselines (autoverificação, segundos, sem GPU) |
| `--fresh` | move as saídas anteriores para uma pasta datada e refaz tudo |
| `--yes` | não pede confirmação (execução sem supervisão) |

O controlador pode ser interrompido com Ctrl+C. Rodando o mesmo comando de novo, ele retoma do
ponto em que parou, sem refazer etapas concluídas.

### O que você precisa instalar antes

| Requisito | Para quê | Como obter |
|---|---|---|
| Python ≥ 3.9 e `curl` | iniciar o controlador e baixar os insumos | já vêm no macOS e na maioria dos Linux |
| Python 3.12, `uv` ou `conda` | criar os ambientes | se não houver nenhum: <https://docs.astral.sh/uv/getting-started/installation/> |
| **Ollama** + modelo `qwen3:8b` | juiz do bloco `gustavo` | <https://ollama.com/download>, depois `ollama pull qwen3:8b` (~5,2 GB). **O controlador não instala o Ollama**; sem ele, só o bloco `gustavo` é bloqueado |
| GPU NVIDIA com 8 GB de VRAM (recomendado) | reproduzir os números | sem ela, roda em MPS ou CPU, com desvios registrados e tempo bem maior |
| ~32 GB livres em disco | ambientes, modelos e saídas | `--check` mostra o necessário e o disponível |

Todo o resto é instalado pelo controlador dentro desta pasta: ambientes em `.reproduce-envs/`,
insumos em `artifacts/` e o llama.cpp b11505. Modelos do Hugging Face vão para o cache definido em
`HF_HOME`; o padrão é `artifacts/hf/`.

### Tempo estimado (GPU NVIDIA)

| Bloco | Tempo | Origem |
|---|---|---|
| `rk` | 21–22 h | medido (RTX 4060 / RX 7600) |
| `davi` | ~1 h | medido (T4 do Colab) |
| `gustavo` | 4–10 h | estimativa, sem medição |

Em MPS e CPU não há medição. Detalhes em [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Instalação manual (sem o controlador)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA
# ou: pip install torch==2.9.1                                                 # macOS / CPU
pip install -r requirements.txt
pip install -r reproduce/requirements/cuda.txt        # só com CUDA (bitsandbytes)
```

A remoção de PII do Davi usa um ambiente separado, porque o Presidio exige `numpy<2.5`:
`reproduce/requirements/pii.txt`. Todas as versões estão fixadas em `reproduce/requirements/`, e as
revisões de modelos e datasets em `reproduce/pins.toml`.

## Variáveis de ambiente

Nenhuma é obrigatória e nenhum segredo é necessário: modelos e dados são públicos. Veja `.env.example`.

| Variável | Uso |
|---|---|
| `HF_HOME` | cache do Hugging Face (padrão: `artifacts/hf/`) |
| `HF_TOKEN` | opcional; só acelera downloads do Hugging Face. Nunca o coloque no repositório |
| `OLLAMA_HOST` | endereço do Ollama (padrão: `http://127.0.0.1:11434`) |
| `LLAMA_SERVER`, `JUDGE_GGUF` | usar um llama-server ou GGUF próprios no juiz do Rubens/Kayque |
| `REPRO_ARTIFACTS`, `REPRO_OUTPUTS`, `REPRO_ENVS` | mudar onde ficam insumos, saídas e ambientes (por exemplo, em outro disco) |

O controlador lê essas variáveis do ambiente em tempo de execução. Ele não lê o `.env` sozinho:
exporte as variáveis no shell (`set -a; source .env; set +a`).

## Dados, pesos e onde ficam

O repositório não contém datasets nem pesos de modelos. Eles ficam fora e são baixados com
verificação de SHA-256:

| Insumo | Onde fica |
|---|---|
| Datasets de fine-tuning do Rubens/Kayque (11), CIMemories, prompts do logit lens | Google Drive público: <https://drive.google.com/drive/folders/1xeiQu2GcCBrb2AdD8SboZfQ3BiLRGrCw>; ids e hashes em `reproduce/manifest.tsv` |
| PrivacyLens (`main_data.json`, 493 cenários) | Hugging Face `SALT-NLP/PrivacyLens`, revisão fixada |
| Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B(-Instruct), Qwen3-8B-GGUF | Hugging Face, revisões fixadas |
| CommonSenseQA | Hugging Face `tau/commonsense_qa`, revisão fixada |
| Datasets do Gustavo (`dataset_degraded_ft.jsonl`, `dataset_safe_ft.jsonl`) | neste repositório (`Gustavo/`), 1,8 MB |

Todos os dados são públicos; não há dado de acesso restrito.

## Saídas que sustentam os números

A reprodução completa custa mais de 25 h de GPU. Por isso, os resultados preliminares estão
versionados em `resultados_preliminares/` (~18 MB):

- `rk/results/`: um JSON por run e benchmark (PrivacyLens, PL semântico, CIMemories com as respostas, CSQA), mais `summary.md` e `timing.md`; `rk/runs/*/train_meta.json`: tempo e loss dos 33 treinos;
- `davi/eval_results/`: PrivacyLens e CSQA por run, com os registros item a item;
- `gustavo/`: `test.jsonl`, pares DPO, exemplos do ICL, julgamentos item a item das 6 condições e `eval/summary.csv`.

`reproduce/baselines/*.csv` resume essas métricas. Para conferir, sem GPU, que esses arquivos
reproduzem a tabela publicada:

```bash
python3 reproduce.py --report-only --preliminary   # gera outputs/preliminar/REPRODUCAO.md
python3 -m unittest discover tests                 # inclui a checagem das 60 células da tabela
```

## Estrutura

```
reproduce.py                 ponto de entrada único
reproduce/                   controlador: aviso, ambientes, download, etapas, comparação, relatório
reproduce/requirements/      dependências fixadas (main, cuda, pii)
reproduce/pins.toml          revisões de modelos/datasets, llama.cpp e hashes
reproduce/manifest.tsv       insumos do Drive com SHA-256
reproduce/baselines/         métricas preliminares em CSV
resultados_preliminares/     saídas preliminares que sustentam os números
RUBENS_KAYQUE/ DAVI/ Gustavo/  código de cada trabalho
docs/PIPELINE.md             pipeline em detalhe, tempos, desvios, solução de problemas
tests/                       testes do controlador (python3 -m unittest discover tests)
```

## Limitações conhecidas

- **Execução completa ainda não validada.** O controlador foi testado com testes unitários e com
  `--check` e `--report-only` num Mac. O `--smoke` e a execução completa em GPU ainda precisam ser
  rodados numa máquina com NVIDIA.
- Resultados idênticos bit a bit não são esperados: as seeds preliminares misturam AMD e NVIDIA, e
  o Davi usou Colab com outra versão do torch. Por isso a comparação usa tolerâncias (ver `docs/PIPELINE.md`).
- Fora da CUDA, o Gustavo roda sem quantização em 4 bits e o Davi treina em float32. Esses desvios
  são registrados no relatório.
