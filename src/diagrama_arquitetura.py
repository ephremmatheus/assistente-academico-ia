"""Desenha o diagrama da arquitetura (figura do artigo e do README).

Uso:
    python -m src.diagrama_arquitetura
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.caminhos import GRAFICOS
from src.graficos import COR_SERIE_1, COR_TEXTO, COR_TEXTO_SECUNDARIO

CAIXAS = {
    # nome: (x, y, título, detalhe)
    "imagem": (0.0, 1.0, "Foto do material", "slide, apostila, livro"),
    "vc": (1.3, 1.0, "Visão Computacional", "pré-processamento + OCR"),
    "ml": (2.6, 1.0, "Machine Learning", "matéria do texto (TF-IDF)"),
    "pergunta": (0.0, 0.0, "Pergunta do aluno", "linguagem natural"),
    "pln": (1.3, 0.0, "PLN", "intenção + busca de trechos"),
    "decisao": (2.6, 0.0, "Agente de decisão", "regras R1–R5"),
    "resposta": (3.9, 0.5, "Resposta", "trecho de origem + avisos"),
}
SETAS = [
    ("imagem", "vc", ""), ("vc", "ml", "texto"), ("pergunta", "pln", ""), ("vc", "pln", "frases"),
    ("pln", "decisao", "relevância"), ("ml", "decisao", "matéria e confiança"),
    ("decisao", "resposta", ""),
]
LARGURA, ALTURA = 0.98, 0.46


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=200)
    for x, y, titulo, detalhe in CAIXAS.values():
        destaque = titulo in {"Visão Computacional", "Machine Learning", "PLN", "Agente de decisão"}
        ax.add_patch(FancyBboxPatch((x - LARGURA / 2, y - ALTURA / 2), LARGURA, ALTURA,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor="#e8f1fc" if destaque else "#f4f4f2",
                                    edgecolor=COR_SERIE_1 if destaque else COR_TEXTO_SECUNDARIO, linewidth=1.2))
        ax.text(x, y + 0.07, titulo, ha="center", va="center", fontsize=10, fontweight="bold", color=COR_TEXTO)
        ax.text(x, y - 0.1, detalhe, ha="center", va="center", fontsize=8, color=COR_TEXTO_SECUNDARIO)

    for origem, destino, rotulo in SETAS:
        x0, y0 = CAIXAS[origem][:2]
        x1, y1 = CAIXAS[destino][:2]
        if y0 == y1:
            inicio, fim = (x0 + LARGURA / 2, y0), (x1 - LARGURA / 2, y1)
        elif x0 == x1:
            inicio, fim = (x0, y0 - ALTURA / 2), (x1, y1 + ALTURA / 2)
        else:
            inicio = (x0 + LARGURA / 2, y0 + (ALTURA / 4 if y1 > y0 else -ALTURA / 4))
            fim = (x1 - LARGURA / 2, y1)
        ax.add_patch(FancyArrowPatch(inicio, fim, arrowstyle="-|>", mutation_scale=12,
                                     color=COR_TEXTO_SECUNDARIO, linewidth=1.1))
        if rotulo:
            if y0 == y1:  # seta horizontal: rótulo centralizado acima
                ax.text((inicio[0] + fim[0]) / 2, y0 + 0.05, rotulo, fontsize=7.5,
                        color=COR_TEXTO_SECUNDARIO, ha="center", va="bottom")
            else:         # seta vertical: rótulo à direita
                ax.text(inicio[0] + 0.04, (inicio[1] + fim[1]) / 2, rotulo, fontsize=7.5,
                        color=COR_TEXTO_SECUNDARIO, ha="left", va="center")

    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(-0.35, 1.35)
    ax.axis("off")
    fig.tight_layout()
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAFICOS / "arquitetura.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
