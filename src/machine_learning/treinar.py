"""Experimento 2: qual classificador identifica melhor a matéria de um texto?

1. Validação cruzada (5 partes) nos parágrafos da Wikipédia. As partes são
   separadas por ARTIGO: parágrafos do mesmo artigo nunca ficam no treino e no
   teste ao mesmo tempo. Se ficassem, o modelo "decoraria" o artigo e a
   avaliação ficaria otimista demais.
2. O melhor modelo é treinado com todos os dados e testado em textos que ele
   nunca viu e que não vieram da Wikipédia:
   - os 12 textos-base do projeto (texto limpo);
   - a saída do OCR nas 60 imagens sintéticas, com e sem pré-processamento.
3. O modelo final é salvo em modelos/classificador_materia.joblib.

Uso:
    python -m src.machine_learning.treinar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.caminhos import GRAFICOS, IMAGENS_SINTETICAS, RESULTADOS
from src.graficos import COR_SERIE_1, COR_TEXTO, COR_TEXTO_SECUNDARIO, MAPA_SEQUENCIAL, estilizar_eixos
from src.visao_computacional.gerar_imagens_sinteticas import carregar_textos

from .classificador import MODELOS, carregar_dados_treino, criar_modelo, salvar


def validacao_cruzada(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    divisor = GroupKFold(n_splits=5)
    linhas, previsoes = [], {}
    for nome in MODELOS:
        acuracias, f1s = [], []
        previsto = np.empty(len(df), dtype=object)
        for treino, teste in divisor.split(df["texto"], df["materia"], groups=df["artigo"]):
            modelo = criar_modelo(nome)
            modelo.fit(df["texto"].iloc[treino], df["materia"].iloc[treino])
            saida = modelo.predict(df["texto"].iloc[teste])
            previsto[teste] = saida
            acuracias.append(accuracy_score(df["materia"].iloc[teste], saida))
            f1s.append(f1_score(df["materia"].iloc[teste], saida, average="macro"))
        previsoes[nome] = previsto
        linhas.append({
            "modelo": nome,
            "acuracia_media": np.mean(acuracias), "acuracia_desvio": np.std(acuracias),
            "f1_macro_medio": np.mean(f1s), "f1_macro_desvio": np.std(f1s),
        })
        print(f"{nome:35s} acurácia {np.mean(acuracias):.3f} ± {np.std(acuracias):.3f} | "
              f"F1-macro {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    return pd.DataFrame(linhas), previsoes


def textos_externos() -> dict[str, pd.DataFrame]:
    """Conjuntos de teste que não vieram da Wikipédia."""
    conjuntos = {"Textos-base (limpos)": pd.DataFrame(carregar_textos())[["materia", "texto"]]}

    caminho_ocr = RESULTADOS / "ocr_easyocr_por_imagem.csv"
    metadados = IMAGENS_SINTETICAS / "metadados.csv"
    if caminho_ocr.exists() and metadados.exists():
        ocr = pd.read_csv(caminho_ocr, encoding="utf-8", keep_default_na=False)
        ocr = ocr.merge(pd.read_csv(metadados)[["arquivo", "materia"]], on="arquivo")
        for configuracao, nome in [("Sem pré-processamento", "OCR sem pré-processamento"),
                                   ("Completo (+ remoção de ruído)", "OCR com pré-processamento")]:
            parte = ocr[ocr["configuracao"] == configuracao]
            conjuntos[nome] = parte.rename(columns={"texto_ocr": "texto"})[["materia", "texto"]]
    return conjuntos


def grafico_comparacao(resultado: pd.DataFrame, destino) -> None:
    dados = resultado.iloc[::-1]
    valores = dados["f1_macro_medio"] * 100
    erros = dados["f1_macro_desvio"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
    ax.barh(dados["modelo"], valores, height=0.55, color=COR_SERIE_1)
    ax.errorbar(valores, dados["modelo"], xerr=erros, fmt="none", ecolor=COR_TEXTO_SECUNDARIO,
                elinewidth=1, capsize=3)
    for y, (v, e) in enumerate(zip(valores, erros)):
        ax.text(v + e + 0.5, y, f"{v:.1f}%", va="center", fontsize=9, color=COR_TEXTO)
    estilizar_eixos(ax)
    ax.tick_params(axis="y", labelcolor=COR_TEXTO, labelsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1-macro (%) na validação cruzada por artigo · maior é melhor",
                  color=COR_TEXTO_SECUNDARIO, fontsize=9)
    ax.set_title("Comparação dos classificadores de matéria", loc="left", fontsize=10, color=COR_TEXTO)
    fig.tight_layout()
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


def grafico_matriz_confusao(reais, previstos, rotulos, titulo: str, destino) -> None:
    matriz = confusion_matrix(reais, previstos, labels=rotulos)
    proporcao = matriz / matriz.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    ax.imshow(proporcao, cmap=MAPA_SEQUENCIAL, vmin=0, vmax=1)
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            cor = "white" if proporcao[i, j] > 0.55 else COR_TEXTO
            ax.text(j, i, matriz[i, j], ha="center", va="center", fontsize=9, color=cor)
    ax.set_xticks(range(len(rotulos)), rotulos, rotation=35, ha="right", fontsize=8.5, color=COR_TEXTO)
    ax.set_yticks(range(len(rotulos)), rotulos, fontsize=8.5, color=COR_TEXTO)
    ax.set_xlabel("Matéria prevista", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("Matéria real", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=10, color=COR_TEXTO)
    fig.tight_layout()
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


def main() -> None:
    df = carregar_dados_treino()
    print(f"{len(df)} parágrafos de {df['artigo'].nunique()} artigos")
    print(df["materia"].value_counts().to_string(), "\n")

    resultado, previsoes = validacao_cruzada(df)
    resultado = resultado.sort_values("f1_macro_medio", ascending=False).reset_index(drop=True)
    melhor = resultado["modelo"].iloc[0]
    print(f"\nMelhor modelo: {melhor}")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(RESULTADOS / "ml_validacao_cruzada.csv", index=False, float_format="%.4f")
    grafico_comparacao(resultado, GRAFICOS / "ml_comparacao_modelos.png")
    rotulos = sorted(df["materia"].unique())
    grafico_matriz_confusao(df["materia"], previsoes[melhor], rotulos,
                            f"Matriz de confusão: {melhor}", GRAFICOS / "ml_matriz_confusao.png")

    # Modelo final: todos os dados, com probabilidades (usadas pelo agente de decisão).
    final = criar_modelo(melhor, com_probabilidade=True)
    final.fit(df["texto"], df["materia"])
    salvar(final, melhor)

    linhas = []
    for nome, conjunto in textos_externos().items():
        previsto = final.predict(conjunto["texto"])
        linhas.append({
            "conjunto": nome, "textos": len(conjunto),
            "acuracia": accuracy_score(conjunto["materia"], previsto),
            "f1_macro": f1_score(conjunto["materia"], previsto, average="macro"),
        })
    externos = pd.DataFrame(linhas)
    externos.to_csv(RESULTADOS / "ml_textos_externos.csv", index=False, float_format="%.4f")
    print("\nModelo final em textos fora da Wikipédia:")
    print(externos.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
