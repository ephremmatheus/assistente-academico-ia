"""Experimento 5: o sistema com fotos reais tiradas com celular.

Para cada foto em `dados/imagens_reais/`:
- identifica a página fotografada e a condição pelo nome do arquivo
  (ex.: p2_inclinada.jpg -> página 2, condição "inclinada"). Se o nome não
  seguir esse padrão, a página é identificada pelo texto lido (menor CER);
- roda o OCR sem e com pré-processamento e mede CER e WER contra o texto impresso;
- classifica a matéria do texto lido e confere com a matéria da página.

Uso:
    python -m src.experimento_fotos_reais
"""

import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.caminhos import DADOS, GRAFICOS, IMAGENS_REAIS, RESULTADOS
from src.graficos import COR_SERIE_1, COR_SERIE_2, COR_TEXTO, COR_TEXTO_SECUNDARIO, estilizar_eixos
from src.machine_learning.classificador import ClassificadorMateria
from src.visao_computacional.avaliacao import cer, wer
from src.visao_computacional.io_imagem import carregar_imagem
from src.visao_computacional.ocr import criar_motor, ler_imagem

PASTA_PAGINAS = DADOS / "material_impressao"
CONFIGURACOES = {"Sem pré-processamento": False, "Com pré-processamento": True}


def gabaritos() -> dict[int, str]:
    return {int(p.stem.split("_")[1]): p.read_text(encoding="utf-8")
            for p in sorted(PASTA_PAGINAS.glob("pagina_*.txt"))}


def materias_das_paginas() -> dict[int, str]:
    tabela = pd.read_csv(PASTA_PAGINAS / "paginas.csv", encoding="utf-8")
    return dict(zip(tabela["pagina"], tabela["materia"]))


def identificar(nome: str, texto_lido: str, textos: dict[int, str]) -> tuple[int, str]:
    padrao = re.match(r"p(\d+)_([a-z]+)", nome.lower())
    if padrao and int(padrao.group(1)) in textos:
        return int(padrao.group(1)), padrao.group(2)
    pagina = min(textos, key=lambda n: cer(textos[n], texto_lido))
    return pagina, "desconhecida"


def main() -> None:
    fotos = sorted(p for p in IMAGENS_REAIS.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not fotos:
        raise SystemExit(f"Nenhuma foto encontrada em {IMAGENS_REAIS}")
    textos, materias = gabaritos(), materias_das_paginas()
    motor, classificador = criar_motor("easyocr"), ClassificadorMateria()

    linhas = []
    for n, foto in enumerate(fotos, start=1):
        imagem = carregar_imagem(foto)
        pagina = condicao = None
        for nome_config, usar in CONFIGURACOES.items():
            inicio = time.perf_counter()
            resultado, _ = ler_imagem(imagem, motor, usar_preprocessamento=usar)
            duracao = time.perf_counter() - inicio
            if pagina is None:
                pagina, condicao = identificar(foto.stem, resultado.texto, textos)
            materia_prevista, probabilidade, _ = classificador.prever(resultado.texto or " ")
            linhas.append({
                "arquivo": foto.name, "pagina": pagina, "condicao": condicao, "configuracao": nome_config,
                "cer": cer(textos[pagina], resultado.texto), "wer": wer(textos[pagina], resultado.texto),
                "confianca_ocr": resultado.confianca_media, "materia_real": materias[pagina],
                "materia_prevista": materia_prevista, "probabilidade_materia": probabilidade,
                "materia_correta": materia_prevista == materias[pagina], "tempo_s": duracao,
                "texto_ocr": resultado.texto,
            })
        ultimas = linhas[-2:]
        print(f"[{n}/{len(fotos)}] {foto.name} (página {pagina}, {condicao}) "
              f"CER sem/com: {ultimas[0]['cer']:.3f} / {ultimas[1]['cer']:.3f}", flush=True)

    df = pd.DataFrame(linhas)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTADOS / "fotos_reais_por_imagem.csv", index=False, encoding="utf-8")

    resumo = df.groupby(["condicao", "configuracao"], sort=False).agg(
        fotos=("arquivo", "count"), cer_medio=("cer", "mean"), wer_medio=("wer", "mean"),
        confianca_ocr=("confianca_ocr", "mean"), materia_correta=("materia_correta", "mean"),
    ).reset_index()
    geral = df.groupby("configuracao", sort=False).agg(
        fotos=("arquivo", "count"), cer_medio=("cer", "mean"), wer_medio=("wer", "mean"),
        confianca_ocr=("confianca_ocr", "mean"), materia_correta=("materia_correta", "mean"),
        tempo_medio_s=("tempo_s", "mean"),
    ).reset_index()
    resumo.to_csv(RESULTADOS / "fotos_reais_por_condicao.csv", index=False, float_format="%.4f")
    geral.to_csv(RESULTADOS / "fotos_reais_resumo.csv", index=False, float_format="%.4f")
    grafico_por_condicao(df, GRAFICOS / "fotos_reais_cer_por_condicao.png")

    print("\nPor condição:")
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nGeral:")
    print(geral.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def grafico_por_condicao(df: pd.DataFrame, destino) -> None:
    ordem = [c for c in ["boa", "sombra", "inclinada", "poucaluz", "dificil", "desconhecida"]
             if c in set(df["condicao"])]
    medias = df.groupby(["condicao", "configuracao"])["cer"].mean().unstack()
    y = np.arange(len(ordem))[::-1]
    altura = 0.36
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(ordem) + 1.4), dpi=200)
    for deslocamento, (config, cor) in zip([altura / 2, -altura / 2],
                                           [("Sem pré-processamento", COR_SERIE_1),
                                            ("Com pré-processamento", COR_SERIE_2)]):
        valores = [medias.loc[c, config] * 100 for c in ordem]
        ax.barh(y + deslocamento, valores, height=altura - 0.04, color=cor, label=config)
        for yi, v in zip(y + deslocamento, valores):
            ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=8, color=COR_TEXTO)
    estilizar_eixos(ax)
    ax.set_yticks(y, ordem, fontsize=9, color=COR_TEXTO)
    ax.set_xlabel("CER médio (%) · menor é melhor", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    ax.set_title(f"OCR em fotos reais por condição ({df['arquivo'].nunique()} fotos)", loc="left",
                 fontsize=10, color=COR_TEXTO)
    legenda = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for texto in legenda.get_texts():
        texto.set_color(COR_TEXTO)
    fig.tight_layout()
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
