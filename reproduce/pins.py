"""Leitor mínimo de pins.toml (seções e pares `chave = "valor"`), compatível com Python 3.9."""
from __future__ import annotations

import re
from pathlib import Path

PINS = Path(__file__).resolve().parent / "pins.toml"
_LINE = re.compile(r'^\s*(?:"([^"]+)"|([A-Za-z0-9_.-]+))\s*=\s*"([^"]*)"\s*$')


def load(path: Path = PINS) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    section = None
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            out[section] = {}
            continue
        m = _LINE.match(line)
        if not m or section is None:
            raise ValueError(f"{path}:{n}: linha inválida: {raw!r}")
        out[section][m.group(1) or m.group(2)] = m.group(3)
    return out
