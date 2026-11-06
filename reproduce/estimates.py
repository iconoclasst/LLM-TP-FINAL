"""Tempo e disco estimados por bloco. Indica sempre o que foi medido e o que é estimativa."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Estimate:
    hours_low: float | None
    hours_high: float | None
    measured: bool
    note: str


# Só há medições em GPU NVIDIA/AMD (RTX 4060, RX 7600) e na T4 do Colab.
CUDA = {
    "rk": Estimate(21, 22, True, "medido: 16,9 h de treino e avaliação + 4–5 h de juiz (RTX 4060 / RX 7600)"),
    "davi": Estimate(0.7, 1.0, True, "medido na T4 do Colab: ~6 min por treino, ~10 min de PrivacyLens por run"),
    "gustavo": Estimate(4, 10, False, "estimativa grosseira, sem medição: 2 treinos DPO, ICL e juiz em 6 condições"),
}


def estimate(block: str, device: str, smoke: bool) -> Estimate:
    if smoke:
        return Estimate(None, None, False, "não medido; poucos minutos por bloco depois dos downloads")
    if device == "cuda":
        return CUDA[block]
    return Estimate(None, None, False, f"sem medição em {device.upper()}; espere bem mais que em GPU NVIDIA")


# Disco em GB (estimado). Componentes compartilhados entre blocos contam uma vez só.
COMPONENTS = {
    "env_main": (6.0, "ambiente Python principal (torch, transformers, ...)"),
    "env_pii": (2.5, "ambiente Presidio/spaCy com en_core_web_lg"),
    "qwen15": (3.1, "Qwen2.5-1.5B-Instruct"),
    "qwen05": (2.0, "Qwen2.5-0.5B e 0.5B-Instruct"),
    "gguf": (5.0, "Qwen3-8B-Q4_K_M.gguf (juiz do Rubens/Kayque)"),
    "llama_cpp": (0.1, "llama.cpp b11505"),
    "out_rk": (3.5, "33 adapters LoRA e resultados"),
    "out_davi": (0.5, "3 adapters LoRA, datasets gerados e resultados"),
    "out_gustavo": (1.5, "2 adapters DPO com checkpoints e resultados"),
    "ollama_qwen3": (5.2, "modelo qwen3:8b do Ollama (juiz do Gustavo)"),
}
NEEDS = {
    "rk": ["env_main", "qwen15", "gguf", "llama_cpp", "out_rk"],
    "davi": ["env_main", "env_pii", "qwen15", "qwen05", "out_davi"],
    "gustavo": ["env_main", "qwen15", "out_gustavo", "ollama_qwen3"],
}


def components(blocks, ollama_model_present: bool) -> list[str]:
    out = []
    for b in blocks:
        for c in NEEDS[b]:
            if c not in out and not (c == "ollama_qwen3" and ollama_model_present):
                out.append(c)
    return out


def disk_needed_gb(blocks, ollama_model_present: bool) -> float:
    return round(sum(COMPONENTS[c][0] for c in components(blocks, ollama_model_present)) * 1.1, 1)
