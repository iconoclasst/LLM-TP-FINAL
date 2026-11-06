"""Caminhos usados pelo controlador. Podem ser trocados por variáveis de ambiente."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "reproduce"
ARTIFACTS = Path(os.environ.get("REPRO_ARTIFACTS", REPO / "artifacts"))   # insumos baixados
OUTPUTS = Path(os.environ.get("REPRO_OUTPUTS", REPO / "outputs"))         # resultados novos
ENVS = Path(os.environ.get("REPRO_ENVS", REPO / ".reproduce-envs"))       # ambientes virtuais
