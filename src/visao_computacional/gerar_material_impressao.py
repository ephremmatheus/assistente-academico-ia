"""Gera um PDF para imprimir e fotografar, com o gabarito de cada página.

Cada página A4 traz uma matéria com seus dois textos de `dados/textos_base.json`.
Como o texto impresso sai daqui, o gabarito das fotos reais não precisa ser
digitado à mão: basta associar cada foto à página fotografada.

Uso:
    python -m src.visao_computacional.gerar_material_impressao
"""

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from src.caminhos import DADOS

from .gerar_imagens_sinteticas import carregar_fonte, carregar_textos, quebrar_linhas

SAIDA = DADOS / "material_impressao"
DPI = 150
LARGURA, ALTURA = 1240, 1754  # A4 a 150 DPI
MARGEM = 130


def desenhar_pagina(numero: int, materia: str, itens: list[dict], fonte_corpo: str) -> tuple[Image.Image, str]:
    pagina = Image.new("L", (LARGURA, ALTURA), 255)
    desenho = ImageDraw.Draw(pagina)
    largura_util = LARGURA - 2 * MARGEM
    y = MARGEM
    linhas_gabarito = []

    def escrever(texto: str, fonte, espacamento: float, espaco_depois: int) -> None:
        nonlocal y
        for linha in quebrar_linhas(texto, fonte, largura_util):
            desenho.text((MARGEM, y), linha, font=fonte, fill=0)
            linhas_gabarito.append(linha)
            y += int(fonte.size * espacamento)
        y += espaco_depois

    escrever(materia, carregar_fonte("DejaVuSans-Bold.ttf", 44), 1.3, 30)
    for item in itens:
        escrever(item["titulo"], carregar_fonte("DejaVuSans-Bold.ttf", 34), 1.4, 10)
        escrever(item["texto"], carregar_fonte(fonte_corpo, 29), 1.55, 40)

    rodape = f"Página {numero}"
    fonte_rodape = carregar_fonte("DejaVuSans.ttf", 24)
    desenho.text((LARGURA - MARGEM - fonte_rodape.getlength(rodape), ALTURA - MARGEM), rodape,
                 font=fonte_rodape, fill=0)
    linhas_gabarito.append(rodape)
    return pagina, "\n".join(linhas_gabarito)


def gerar(saida: Path = SAIDA) -> Path:
    por_materia: dict[str, list[dict]] = defaultdict(list)
    for item in carregar_textos():
        por_materia[item["materia"]].append(item)

    saida.mkdir(parents=True, exist_ok=True)
    paginas = []
    with (saida / "paginas.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["pagina", "materia"])
        for numero, (materia, itens) in enumerate(por_materia.items(), start=1):
            fonte_corpo = "DejaVuSerif.ttf" if numero % 2 else "DejaVuSans.ttf"
            pagina, gabarito = desenhar_pagina(numero, materia, itens, fonte_corpo)
            (saida / f"pagina_{numero}.txt").write_text(gabarito, encoding="utf-8")
            escritor.writerow([numero, materia])
            paginas.append(pagina)

    pdf = saida / "material_para_imprimir.pdf"
    paginas[0].save(pdf, save_all=True, append_images=paginas[1:], resolution=DPI)
    return pdf


if __name__ == "__main__":
    print(f"PDF gerado em {gerar()}")
