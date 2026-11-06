# Como executar: guia das pastas e passo a passo

Este é o guia rápido. O resumo do projeto está no [README](README.md). Detalhes de cada etapa,
critérios de verificação e solução de problemas estão em [`docs/PIPELINE.md`](docs/PIPELINE.md).

## O que cada pasta faz

| Pasta / arquivo | O que tem |
|---|---|
| `reproduce.py` | **Ponto de entrada único.** Roda os três experimentos em sequência e compara com os resultados preliminares |
| `reproduce/` | Código do controlador: verificação do computador, criação dos ambientes, download dos insumos, execução das etapas, comparação e relatório |
| `reproduce/requirements/` | Dependências com versões fixadas: `main.txt` (ambiente principal), `cuda.txt` (só GPU NVIDIA), `pii.txt` (remoção de PII do Davi) |
| `reproduce/pins.toml` | Revisões fixadas de modelos e datasets do Hugging Face, build do llama.cpp e hashes |
| `reproduce/manifest.tsv` | Arquivos baixados do Google Drive (datasets, CIMemories, prompts), com SHA-256 |
| `reproduce/baselines/` | Métricas dos resultados preliminares em CSV, usadas na comparação |
| `reproduce/steps/` | Pequenos scripts usados pelo controlador: download do HF, llama.cpp e wrappers dos juízes |
| `RUBENS_KAYQUE/reproducao_tabela_resultados/` | Experimento 1: reprodução do *Privacy Collapse* com LoRA no Qwen2.5-1.5B em 11 datasets × 3 seeds, avaliado em PrivacyLens, CIMemories e CSQA. Scripts em `experiments/scripts/` |
| `DAVI/solutions/` | Experimento 2: mitigações no dataset prestativo. `gen_datasets/` gera as versões sem PII e com troca de papéis; `src/fine_tune_models.py` treina; `benchmarks/` avalia (PrivacyLens e CSQA); `data/*/logs.txt` guarda as contagens originais das transformações |
| `DAVI/logit_lens/` | Experimento 2 (análise): logit lens entre Qwen2.5-0.5B e 0.5B-Instruct |
| `Gustavo/` | Experimento 3: `run_all.py` monta os pares e treina o DPO (`collapse` e `control`); `ICL.py` gera as respostas das 3 condições de ICL; `Tester.py` julga as 6 condições com o `qwen3:8b` no Ollama. `make_pairs.py` e `train_dpo.py` são versões anteriores, não usadas pelo controlador |
| `resultados_preliminares/` | Resultados já obtidos pela equipe (JSONs por run, julgamentos, `summary.csv`). Servem de evidência dos números e de baseline |
| `Descricoes_experimentos/` | Resumo dos resultados do experimento 1 (PDF) |
| `docs/PIPELINE.md` | Pipeline em detalhe: etapas, tolerâncias, tempos, desvios e solução de problemas |
| `docs/superpowers/` | Design e plano de implementação do controlador |
| `tests/` | Testes do controlador (`python3 -m unittest discover tests`); não usam GPU nem baixam modelos |
| `requirements.txt` | Dependências do ambiente principal, para quem quiser montá-lo à mão |
| `.env.example` | Variáveis de ambiente opcionais |

Criadas pela execução, fora do Git: `artifacts/` (insumos baixados), `.reproduce-envs/` (ambientes
Python) e `outputs/` (resultados novos e relatório).

## Passo a passo para executar todos os experimentos

Recomendado: uma máquina com GPU NVIDIA (8 GB de VRAM), cerca de 32 GB livres em disco e conexão
com a internet.

**1. Baixe o repositório e entre na branch.**

```bash
git clone https://github.com/iconoclasst/LLM-TP-FINAL.git
cd LLM-TP-FINAL
git checkout feature/controlador-reproducao
```

**2. Verifique o computador.** Este comando não instala nem executa nada:

```bash
python3 reproduce.py --check
```

Ele lista os **BLOQUEIOS**, os **AVISOS**, o que será instalado e baixado, o tempo estimado e o
espaço em disco. Resolva os bloqueios indicados antes de continuar.

**3. Instale o Ollama e o modelo do juiz do Gustavo.** É a única instalação manual:

```bash
# baixe e instale: https://ollama.com/download
ollama serve            # ou deixe o aplicativo do Ollama aberto
ollama pull qwen3:8b    # ~5,2 GB
```

Sem o Ollama, os blocos do Rubens/Kayque e do Davi rodam normalmente; só o do Gustavo é pulado.

**4. Faça um teste rápido do pipeline.** Use poucos itens; leva alguns minutos depois dos downloads:

```bash
python3 reproduce.py --smoke
```

Responda `s` quando ele perguntar `Continuar? [s/N]`. Depois confira
`outputs/smoke/REPRODUCAO.md`: todas as etapas precisam estar `ok`, e o veredito deve ser `SMOKE`.

**5. Rode todos os experimentos.** Na GPU, são mais de 25 h:

```bash
python3 reproduce.py
```

- Para rodar sem supervisão, sem a pergunta de confirmação: `python3 reproduce.py --yes`.
- Pode interromper com Ctrl+C. Rodando o mesmo comando de novo, ele continua do ponto em que parou.
- Para rodar só um experimento: `--only rk`, `--only davi` ou `--only gustavo`.

**6. Veja o resultado.**

- `outputs/REPRODUCAO.md`: veredito geral (`REPRODUZIDO`, `PARCIAL` ou `NAO_REPRODUZIDO`), conclusões dos relatórios, verificações exatas e a tabela de cada métrica, novo × preliminar;
- `outputs/comparison.csv`: a mesma comparação em CSV;
- `outputs/environment.json`: sistema, GPU e versões de todos os pacotes;
- `outputs/rk/`, `outputs/davi/` e `outputs/gustavo/`: resultados completos de cada experimento;
- `outputs/logs/`: um log por etapa, útil para investigar falhas.

Para refazer só o relatório: `python3 reproduce.py --report-only`.

## Conferir os números sem GPU

Os resultados preliminares estão versionados. Para conferir, em segundos, que eles reproduzem as
tabelas dos relatórios:

```bash
python3 reproduce.py --report-only --preliminary   # gera outputs/preliminar/REPRODUCAO.md
python3 -m unittest discover tests
```
