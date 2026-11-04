# Controlador único de reprodução: design

Status: aguardando aprovação. Branch: `feature/controlador-reproducao`.

## 1. Intenção

Hoje o repositório são 3 trabalhos separados (Rubens/Kayque, Davi, Gustavo). O objetivo é que **um único
comando** execute os experimentos, um de cada vez, e produza arquivos de resultado que permitam
**conferir os resultados preliminares e demonstrar a reprodutibilidade**. A entrega final do TP é em 13/11/2026, e a
especificação exige comandos exatos, versões fixadas e saídas salvas.

## 2. Objetivos, não objetivos e o que seria fracasso

**Objetivos**
- `python3 reproduce.py` faz tudo: prepara ambiente, baixa insumos, roda, compara e gera o relatório.
- Usa GPU se houver (CUDA, depois MPS); senão, CPU.
- Pode ser interrompido e retomado sem perder o que já terminou.
- Avisa antes de começar o que está faltando, em vez de falhar horas depois.
- `docs/PIPELINE.md` explica o pipeline e traz estimativas de tempo.

**Não objetivos**
- Corrigir a metodologia dos experimentos. O vazamento treino/teste do Gustavo é **reproduzido e reportado**, não corrigido.
- Unificar as métricas dos três trabalhos numa só tabela.
- Tornar o Colab ou o PowerShell parte do fluxo.

**Fracasso, mesmo que tecnicamente funcione**
- Marcar "reproduzido" para algo que não foi comparado (por exemplo, sem baseline).
- Sobrescrever os resultados preliminares com os novos.
- Alterar hiperparâmetro, prompt ou métrica ao adaptar os scripts.

## 3. O que "mesmos resultados" significa

Números idênticos não são garantidos entre máquinas: as 3 seeds preliminares já misturam AMD e NVIDIA, e
Davi usou Colab T4 com outra versão do torch. Por isso há três níveis de verificação:

| Nível | O que verifica | Critério |
|---|---|---|
| Insumos | SHA-256 de datasets e benchmarks; contagens das transformações do Davi (1.729 trocas; 1.261 linhas, 1.413 PERSON e 54 LOCATION) | Igualdade exata |
| Métricas | Cada métrica de cada run contra o baseline | `|novo − baseline| ≤ max(piso, 2·dp das seeds)` |
| Conclusões | Afirmações do relatório, como `synth_helpful < synth_control` nas 3 seeds e todos os fine-tunes abaixo do base | Verdadeiro ou falso |

Pisos, em pontos percentuais: logprob 10; amostrado 5; semântico 5; CIMemories 10; CSQA 2.
Os valores são escolha do design, baseados nos desvios observados e no limite de ~10 pontos do CIMemories citado no resumo de resultados.

Vereditos: `OK`, `FORA_DA_TOLERANCIA`, `SEM_BASELINE`, `NAO_EXECUTADO`. Nenhum resultado sem comparação conta como `OK`.

## 4. O que é executado

| Bloco | Conteúdo | Baseline |
|---|---|---|
| `rk` | Modelo base + 11 datasets × 3 seeds: LoRA, PrivacyLens, CSQA, CIMemories, juiz e PL semântico, tabela final | `results/` do Drive |
| `davi` | Gera `nopii` e `role_swap_v2`; 3 fine-tunes (`degraded`, `nopii_ft`, `role_swap_ft_v2`, seed 0); PrivacyLens nos 3; CSQA nos 3 e no base (`untuned`); logit lens | `eval_results/` do Drive; logit lens sem baseline |
| `gustavo` | Pares DPO, 2 treinos DPO, 3 condições ICL, avaliação com juiz (`LIMIT=None`) | `summary.csv`, quando chegar |

Fica de fora, por não ter resultado preliminar: `role_swap_ft` v1, run `teste`, `cand_*`,
`_q15__base_v1_sem_schema`, `privacy_refine.py` e juiz/CIMemories do Davi. Estes dois últimos poderão ser ligados por
`--extend`, marcados como `SEM_BASELINE`.

## 5. Interface

```
python3 reproduce.py                  # tudo
python3 reproduce.py --check          # só verifica ambiente, insumos e disco; mostra plano e tempo estimado
python3 reproduce.py --only rk        # um bloco: rk, davi ou gustavo
python3 reproduce.py --smoke          # poucos itens e passos, para testar o pipeline; nunca compara com baseline
python3 reproduce.py --report-only    # só refaz comparação e relatório
python3 reproduce.py --fresh          # ignora o que já foi concluído
python3 reproduce.py --extend         # inclui juiz e CIMemories do Davi
```

## 6. Etapas do controlador

1. **Bootstrap.** O script usa só a biblioteca padrão, acha ou cria um Python 3.12 (`python3.12`, `uv` ou `conda`), monta os ambientes virtuais e se reexecuta dentro deles.
2. **Preflight.** Dispositivo, disco, Ollama, insumos e versões. Termina com a lista do que falta e como obter.
3. **Download.** Drive e Hugging Face, com verificação de hash.
4. **Execução.** Blocos em sequência (`rk`, `davi`, `gustavo`), um processo por etapa, com GPU liberada entre etapas. Cada etapa só conta como concluída com saída válida e é registrada em `ledger.jsonl`.
5. **Comparação e relatório.** Gera os artefatos da seção 8.

Se uma etapa falhar, o bloco é interrompido, os demais continuam e o relatório mostra a falha. O código de saída fica diferente de zero.

## 7. Alterações em código e arquivos

### Novo
```
reproduce.py                 ponto de entrada
reproduce/                   bootstrap, preflight, download, etapas, comparação, relatório
reproduce/manifest.tsv       id no Drive, caminho e SHA-256
reproduce/pins.toml          revisões de modelos e datasets, build do llama.cpp
reproduce/requirements/      dependências fixadas por ambiente e plataforma
reproduce/baselines/         métricas preliminares em CSV, versionadas
docs/PIPELINE.md             pipeline e estimativas
README.md                    resumo do projeto e comando único
```
Dados baixados em `artifacts/` e saídas em `outputs/<bloco>/`, ambos fora do Git.

### Nos scripts existentes: só dispositivo, caminhos e parametrização
Todos passam a ler dispositivo e caminhos do controlador, mantendo os valores atuais como padrão.

- **Rubens/Kayque:** `common.py` com caminhos configuráveis; `device_map`, `torch.cuda.*` e geradores fixos em CUDA passam a seguir o dispositivo; orçamento de memória por dispositivo; `run_judge.py` sem caminho do Windows.
- **Davi:** o `assert` de CUDA vira detecção; `GradScaler` e `autocast` por dispositivo; `torch.cuda.*` protegidos; caminhos passados por argumento; `logit_lens.py` recebe caminhos de entrada e saída por argumento.
- **Gustavo:** quantização em 4 bits só com CUDA; `bf16` condicional; `LIMIT` por argumento (padrão: tudo); caminhos configuráveis; seed passada ao Ollama; opção de reaproveitar pares do Drive.

Hiperparâmetros, prompts e métricas não mudam. Cada desvio forçado pelo dispositivo fica registrado nos metadados e no relatório.

## 8. Saídas

```
outputs/REPRODUCAO.md        veredito geral e por experimento, tabela novo × baseline, desvios
outputs/comparison.csv       uma linha por run e métrica
outputs/environment.json     SO, GPU, versões, revisões, commit do Git
outputs/ledger.jsonl         etapas, duração e código de saída
outputs/rk/ davi/ gustavo/   resultados completos de cada experimento
```

## 9. Insumos

| Insumo | Origem | Situação |
|---|---|---|
| CIMemories (490 cenários, usa os 100 primeiros) | Drive | OK |
| `experiments/data/*.jsonl` | Drive | OK |
| Resultados de Rubens/Kayque | Drive | OK; `summary.md` igual ao versionado, exceto quebras de linha |
| Resultados e prompts do Davi (inclui `untuned`) | Drive | OK |
| PrivacyLens | Hugging Face `SALT-NLP/PrivacyLens` | Disponível; revisão a fixar |
| Qwen2.5-1.5B-Instruct, 0.5B, 0.5B-Instruct, Qwen3-8B Q4_K_M | Hugging Face | Disponíveis; revisões a fixar |
| llama.cpp b11505 | GitHub | Há builds para macOS, Linux e Windows |
| Ollama + `llama3.2:3b` + `qwen3:8b` | Instalação local | **Falta nesta máquina**; o preflight orienta |
| `train_meta.json` das 33 runs | Drive | **Só 1 (dolly s2)**; colunas de tempo e loss do baseline vêm de `summary.json` |
| Pares DPO, `test.jsonl`, `summary.csv` do Gustavo | Drive | **Pendente**; o controlador baixa assim que aparecerem e aceita qualquer estrutura de subpastas |
| `dataset_nopii_ft`, `dataset_role_swap_ft_v2` | Gerados | Não estão no Drive; regenerados e conferidos pelas contagens dos logs |

Sem os arquivos do Gustavo o bloco ainda roda: gera os pares localmente com seed e marca `SEM_BASELINE`.

## 10. Restrições e riscos

- **Testes nesta máquina:** o Mac M4 de 16 GB permite testar `--smoke` em MPS e CPU. **O caminho CUDA não pode ser testado aqui** e precisa de um `--smoke` na máquina com NVIDIA.
- **Desvio obrigatório em MPS/CPU:** o Gustavo roda sem quantização em 4 bits, pois `bitsandbytes` depende de CUDA na prática. O relatório registra isso.
- **Disco:** cerca de 22 GB de modelos, ambientes e adapters, com 28 GB livres nesta máquina. O preflight exige folga e avisa antes de iniciar.
- **Versões:** o baseline do Davi usou torch 2.11 sem versões fixadas; o do Rubens/Kayque, torch 2.9.1 e `transformers` 5.19. O controlador usa o lock de Rubens/Kayque. Presidio/spaCy e `trl` podem exigir ambientes próprios; isso será definido na implementação.
- **Gustavo:** não há registro das versões usadas. O modelo das contrapartes segue o código (`llama3.2:3b`) até confirmação em contrário.
- **Drive:** a reprodução depende de o link continuar público; as métricas ficam também em `reproduce/baselines/`.

## 11. Estimativas de tempo

| Bloco | RTX 4060 / T4 | Base |
|---|---|---|
| `rk` | 21–22 h (16,9 h + 4–5 h de juiz) | Medido no `ledger.jsonl` e no relatório |
| `davi` | ~50 min (T4) | Medido: treino ~6 min por run; PrivacyLens ~10 min; CSQA ~25 s |
| `gustavo` | 12–25 h | **Estimativa grosseira, sem medição**; pares reaproveitados poupam 6–10 h |

MPS e CPU não têm medição. Na implementação, um teste curto medirá o fator nesta máquina, e o `--check` usará o fator
da máquina atual. O `PIPELINE.md` indicará claramente o que é medido e o que é estimado.

## 12. Critérios de aceitação

- `--report-only` com os resultados preliminares reproduz os valores de `summary_original.md`.
- `--check` lista corretamente o que falta.
- `--smoke` completa os 3 blocos no Mac, exceto o que depender de Ollama ausente, que é reportado.
- Interromper e retomar não refaz etapas concluídas.
- Nada é commitado na `main`.

## 13. Decidido pelo usuário e deixado ao implementador

**Decidido pelo usuário:** um comando único externo às pastas; usar GPU se houver; estimativas no MD; scripts podem ser alterados, em
branch fora da `main`; implementar considerando que os arquivos do Gustavo vão existir.

**Escolhas do design (revisar):** tolerâncias da seção 3; escopo da seção 4; `--extend` desligado por padrão; ambiente baseado no lock de Rubens/Kayque; manter o vazamento do Gustavo.

**Deixado ao implementador:** organização interna de `reproduce/`, formato dos logs e escolha de `trl` compatível.
