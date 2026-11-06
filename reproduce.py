#!/usr/bin/env python3
"""Ponto de entrada único: `python3 reproduce.py --help`. Veja README.md e docs/PIPELINE.md."""
import sys

if sys.version_info < (3, 9):
    sys.exit("Este script precisa de Python 3.9 ou mais novo para iniciar (ele mesmo obtém o 3.12 depois).")

from reproduce.cli import main

if __name__ == "__main__":
    sys.exit(main())
