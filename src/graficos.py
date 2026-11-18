"""Estilo comum dos gráficos dos experimentos (cores, eixos e mapa de cor sequencial)."""

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

COR_SERIE_1 = "#2a78d6"
COR_SERIE_2 = "#eb6834"
COR_SERIE_3 = "#1baf7a"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e4e3df"

# Escala sequencial de uma cor só (claro -> escuro), para matrizes de confusão.
MAPA_SEQUENCIAL = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#86b6ef", "#2a78d6", "#184f95"])


def estilizar_eixos(ax, grade: str = "x") -> None:
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_TEXTO_SECUNDARIO)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO, length=0)
    ax.grid(axis=grade, color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
