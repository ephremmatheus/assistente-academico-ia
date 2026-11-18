"""Gera imagens de texto com degradações controladas, junto com o texto correto (gabarito).

Para medir o OCR é preciso saber o texto certo de cada imagem. Transcrever fotos
à mão é lento, então este script desenha os textos de `dados/textos_base.json`
e aplica defeitos comuns em fotos de celular: baixo contraste, sombra,
inclinação, desfoque, ruído e compressão JPEG. Os parâmetros de cada imagem
ficam em `metadados.csv`, o que permite analisar depois quais defeitos mais
atrapalham a leitura.

As fontes usadas são as que vêm junto com o matplotlib (DejaVu e STIX, de
licença livre e com todos os acentos do português), e a semente aleatória é
fixa. Assim, o mesmo comando gera as mesmas imagens em qualquer computador.

Uso:
    python -m src.visao_computacional.gerar_imagens_sinteticas --quantidade 60 --semente 42
"""

import argparse
import csv
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.caminhos import IMAGENS_SINTETICAS, TEXTOS_BASE

PASTA_FONTES = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
# Sem serifa (comum em slides), com serifa (livros) e monoespaçada (código).
FONTES = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "STIXGeneral.ttf", "DejaVuSansMono.ttf"]


@dataclass
class ParametrosImagem:
    fonte: str
    tamanho_fonte: int
    angulo: float         # graus; positivo = anti-horário
    desfoque: float       # raio do desfoque gaussiano em pixels
    ruido: float          # desvio padrão do ruído gaussiano (escala 0-255)
    sombra: float         # 0 = sem sombra; 0.5 = um lado da página fica 50% mais escuro
    cor_papel: int        # 255 = branco puro
    cor_tinta: int        # 0 = preto puro
    qualidade_jpeg: int   # 100 = sem perda


def carregar_fonte(nome: str, tamanho: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(PASTA_FONTES / nome), tamanho)


def carregar_textos(caminho: Path = TEXTOS_BASE) -> list[dict]:
    return json.loads(caminho.read_text(encoding="utf-8"))


def quebrar_linhas(texto: str, fonte: ImageFont.FreeTypeFont, largura_maxima: int) -> list[str]:
    """Quebra o texto em linhas que cabem na largura, medindo em pixels (não em caracteres).

    Estratégia gulosa: coloca palavras na linha atual enquanto couberem.
    """
    linhas, atual = [], ""
    for palavra in texto.split():
        tentativa = f"{atual} {palavra}".strip()
        if fonte.getlength(tentativa) <= largura_maxima or not atual:
            atual = tentativa
        else:
            linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas


def renderizar_texto(
    texto: str, tamanho_fonte: int = 28, nome_fonte: str = FONTES[0], largura: int = 1200, margem: int = 60
) -> tuple[Image.Image, str]:
    """Desenha o texto preto em fundo branco. Devolve a imagem e o texto como foi quebrado em linhas."""
    fonte = carregar_fonte(nome_fonte, tamanho_fonte)
    linhas = quebrar_linhas(texto, fonte, largura - 2 * margem)
    altura_linha = int(tamanho_fonte * 1.6)
    altura = 2 * margem + altura_linha * len(linhas)
    imagem = Image.new("L", (largura, altura), 255)
    desenho = ImageDraw.Draw(imagem)
    for i, linha in enumerate(linhas):
        desenho.text((margem, margem + i * altura_linha), linha, font=fonte, fill=0)
    return imagem, "\n".join(linhas)


def sortear_parametros(rng: random.Random) -> ParametrosImagem:
    return ParametrosImagem(
        fonte=rng.choice(FONTES),
        tamanho_fonte=rng.randint(22, 34),
        angulo=round(rng.uniform(-6, 6), 2),
        desfoque=round(rng.uniform(0, 1.5), 2),
        ruido=round(rng.uniform(0, 20), 1),
        sombra=round(rng.choice([0, rng.uniform(0.2, 0.6)]), 2),
        cor_papel=rng.randint(190, 255),
        cor_tinta=rng.randint(0, 80),
        qualidade_jpeg=rng.randint(50, 95),
    )


def aplicar_degradacao(imagem: Image.Image, d: ParametrosImagem, rng_np: np.random.Generator) -> Image.Image:
    """Aplica os defeitos em ordem parecida com a de uma foto real.

    A compressão JPEG não é aplicada aqui: ela acontece ao salvar o arquivo.
    """
    pixels = np.asarray(imagem, dtype=np.float32) / 255.0

    # Contraste: o papel não é branco puro e a tinta não é preto puro.
    pixels = d.cor_tinta + pixels * (d.cor_papel - d.cor_tinta)
    resultado = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    resultado = resultado.rotate(d.angulo, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=d.cor_papel)

    # Sombra: escurecimento gradual de um lado da foto até o outro. Aplicada
    # depois da rotação para cobrir a imagem inteira, como a luz de verdade.
    if d.sombra > 0:
        pixels = np.asarray(resultado, dtype=np.float32)
        gradiente = np.linspace(1.0 - d.sombra, 1.0, pixels.shape[1], dtype=np.float32)
        if rng_np.random() < 0.5:
            gradiente = gradiente[::-1]
        resultado = Image.fromarray(np.clip(pixels * gradiente[np.newaxis, :], 0, 255).astype(np.uint8))

    if d.desfoque > 0:
        resultado = resultado.filter(ImageFilter.GaussianBlur(d.desfoque))

    if d.ruido > 0:
        pixels = np.asarray(resultado, dtype=np.float32)
        pixels = pixels + rng_np.normal(0, d.ruido, pixels.shape)
        resultado = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    return resultado


def gerar(quantidade: int, semente: int, saida: Path = IMAGENS_SINTETICAS) -> Path:
    rng = random.Random(semente)
    rng_np = np.random.default_rng(semente)
    textos = carregar_textos()
    saida.mkdir(parents=True, exist_ok=True)

    caminho_metadados = saida / "metadados.csv"
    with caminho_metadados.open("w", newline="", encoding="utf-8") as arquivo_csv:
        campos = ["arquivo", "materia"] + list(ParametrosImagem.__dataclass_fields__)
        escritor = csv.DictWriter(arquivo_csv, fieldnames=campos)
        escritor.writeheader()

        for i in range(quantidade):
            item = textos[i % len(textos)]
            parametros = sortear_parametros(rng)
            limpa, gabarito = renderizar_texto(item["texto"], parametros.tamanho_fonte, parametros.fonte)
            degradada = aplicar_degradacao(limpa, parametros, rng_np)

            nome = f"sintetica_{i:03d}"
            buffer = io.BytesIO()
            degradada.save(buffer, format="JPEG", quality=parametros.qualidade_jpeg)
            (saida / f"{nome}.jpg").write_bytes(buffer.getvalue())
            (saida / f"{nome}.txt").write_text(gabarito, encoding="utf-8")

            escritor.writerow({"arquivo": f"{nome}.jpg", "materia": item["materia"], **asdict(parametros)})

    return saida


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera imagens sintéticas de texto com gabarito.")
    parser.add_argument("--quantidade", type=int, default=60)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--saida", type=Path, default=IMAGENS_SINTETICAS)
    args = parser.parse_args()
    pasta = gerar(args.quantidade, args.semente, args.saida)
    print(f"{args.quantidade} imagens geradas em {pasta}")


if __name__ == "__main__":
    main()
