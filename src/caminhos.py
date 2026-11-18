"""Caminhos usados pelo projeto, todos relativos à raiz do repositório."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DADOS = RAIZ / "dados"
TEXTOS_BASE = DADOS / "textos_base.json"
IMAGENS_SINTETICAS = DADOS / "sinteticas"
IMAGENS_REAIS = DADOS / "imagens_reais"

RESULTADOS = RAIZ / "resultados"
GRAFICOS = RESULTADOS / "graficos"
EXEMPLOS = RESULTADOS / "exemplos"
