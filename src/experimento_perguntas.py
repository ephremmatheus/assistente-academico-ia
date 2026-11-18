"""Experimentos 3 e 4: PLN (intenção e busca) e calibração do agente de decisão.

Usa as perguntas de `dados/perguntas.csv`, feitas sobre as 6 páginas do
material impresso. O texto de cada página pode vir de duas fontes:
- "gabarito": o texto exato da página (avalia o PLN isolado, sem erros de OCR);
- "ocr":      o texto lido pelo OCR da foto real "boa" de cada página
              (avalia o sistema como o usuário o usa). Exige ter rodado antes
              `python -m src.experimento_fotos_reais`.

Métricas:
- Intenção: acurácia das regras.
- Busca: Recall@1, Recall@3 e MRR (posição média inversa do primeiro trecho certo).
- Decisão: para cada limiar de similaridade, % de perguntas respondíveis
  respondidas corretamente e % de perguntas sem resposta recusadas.

Uso:
    python -m src.experimento_perguntas --fonte gabarito
    python -m src.experimento_perguntas --fonte ocr
"""

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.assistente import AssistenteAcademico
from src.caminhos import DADOS, GRAFICOS, RESULTADOS
from src.decisao.agente import AgenteDecisao
from src.graficos import COR_SERIE_1, COR_SERIE_2, COR_TEXTO, COR_TEXTO_SECUNDARIO, estilizar_eixos
from src.pln.busca import METODOS, IndiceBusca
from src.pln.intencao import reconhecer_intencao
from src.pln.texto import normalizar

PASTA_PAGINAS = DADOS / "material_impressao"


def _tokens(texto: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", normalizar(texto)).split()


def contem_resposta(trecho: str, esperada: str, minimo: float = 0.8) -> bool:
    """O trecho contém a resposta esperada? Tolera pequenos erros de OCR:
    basta que 80% das palavras da resposta apareçam no trecho."""
    esperadas = _tokens(esperada)
    presentes = set(_tokens(trecho))
    return bool(esperadas) and sum(t in presentes for t in esperadas) / len(esperadas) >= minimo


def carregar_textos(fonte: str) -> dict[int, str]:
    if fonte == "gabarito":
        return {n: (PASTA_PAGINAS / f"pagina_{n}.txt").read_text(encoding="utf-8") for n in range(1, 7)}
    fotos = pd.read_csv(RESULTADOS / "fotos_reais_por_imagem.csv", encoding="utf-8", keep_default_na=False)
    fotos = fotos[(fotos["condicao"] == "boa") & (fotos["configuracao"] == "Com pré-processamento")]
    return {int(linha.pagina): linha.texto_ocr for linha in fotos.itertuples()}


def avaliar_intencao(perguntas: pd.DataFrame) -> pd.DataFrame:
    perguntas = perguntas.copy()
    perguntas["intencao_prevista"] = [reconhecer_intencao(p).tipo for p in perguntas["pergunta"]]
    acertos = (perguntas["intencao_prevista"] == perguntas["intencao"]).mean()
    print(f"Intenção: acurácia {acertos:.3f} ({(perguntas['intencao_prevista'] == perguntas['intencao']).sum()}"
          f"/{len(perguntas)})")
    print(pd.crosstab(perguntas["intencao"], perguntas["intencao_prevista"],
                      rownames=["real"], colnames=["prevista"]).to_string(), "\n")
    return perguntas


def avaliar_busca(perguntas: pd.DataFrame, textos: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    respondiveis = perguntas[(perguntas["tem_resposta"] == 1) & (perguntas["intencao"] != "resumo")]
    linhas, detalhes = [], []
    for metodo in METODOS:
        indices = {n: IndiceBusca(t, metodo) for n, t in textos.items()}
        posicoes = []
        for p in respondiveis.itertuples():
            if p.pagina not in indices:
                continue
            resultados = indices[p.pagina].buscar(p.pergunta, k=len(indices[p.pagina].frases),
                                                  intencao=reconhecer_intencao(p.pergunta).tipo)
            posicao = next((i + 1 for i, r in enumerate(resultados)
                            if contem_resposta(r.frase.texto, p.resposta_esperada)), None)
            posicoes.append(posicao)
            detalhes.append({"metodo": metodo, "pagina": p.pagina, "pergunta": p.pergunta,
                             "posicao_correta": posicao,
                             "similaridade_top1": resultados[0].similaridade if resultados else 0.0})
        n = len(posicoes)
        linhas.append({
            "metodo": metodo, "perguntas": n,
            "recall_1": sum(1 for x in posicoes if x == 1) / n,
            "recall_3": sum(1 for x in posicoes if x and x <= 3) / n,
            "mrr": sum(1 / x for x in posicoes if x) / n,
        })
    resumo = pd.DataFrame(linhas)
    print("Busca:")
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")
    return resumo, pd.DataFrame(detalhes)


def calibrar_limiar(perguntas: pd.DataFrame, textos: dict[int, str], metodo: str) -> tuple[pd.DataFrame, float]:
    """Varre limiares e mede o equilíbrio entre responder certo e recusar o que não está no material."""
    casos = []
    for p in perguntas[perguntas["intencao"] != "resumo"].itertuples():
        if p.pagina not in textos:
            continue
        resultados = IndiceBusca(textos[p.pagina], metodo).buscar(
            p.pergunta, k=1, intencao=reconhecer_intencao(p.pergunta).tipo)
        similaridade = resultados[0].similaridade if resultados else 0.0
        certo = bool(resultados) and p.tem_resposta == 1 and contem_resposta(resultados[0].frase.texto,
                                                                              p.resposta_esperada)
        casos.append((p.tem_resposta == 1, similaridade, certo))

    linhas = []
    for limiar in np.round(np.arange(0.0, 0.61, 0.01), 2):
        resp = [(s, c) for tem, s, c in casos if tem]
        sem = [s for tem, s, _ in casos if not tem]
        respondidas_certas = sum(1 for s, c in resp if s >= limiar and c) / len(resp)
        recusadas_certas = sum(1 for s in sem if s < limiar) / len(sem) if sem else 0.0
        acerto_total = (sum(1 for s, c in resp if s >= limiar and c) + sum(1 for s in sem if s < limiar)) / len(casos)
        linhas.append({"limiar": limiar, "respondiveis_corretas": respondidas_certas,
                       "sem_resposta_recusadas": recusadas_certas, "acerto_total": acerto_total})
    tabela = pd.DataFrame(linhas)
    melhores = tabela[tabela["acerto_total"] == tabela["acerto_total"].max()]
    escolhido = float(melhores["limiar"].iloc[len(melhores) // 2])  # meio da faixa de melhor acerto
    return tabela, escolhido


def grafico_limiar(tabela: pd.DataFrame, escolhido: float, fonte: str, destino) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    ax.plot(tabela["limiar"], tabela["respondiveis_corretas"] * 100, color=COR_SERIE_1, linewidth=2,
            label="Perguntas com resposta: respondidas corretamente")
    ax.plot(tabela["limiar"], tabela["sem_resposta_recusadas"] * 100, color=COR_SERIE_2, linewidth=2,
            label="Perguntas sem resposta: recusadas")
    ax.axvline(escolhido, color=COR_TEXTO_SECUNDARIO, linewidth=1, linestyle="--")
    ax.text(escolhido + 0.005, 4, f"limiar escolhido = {escolhido:.2f}", fontsize=8.5, color=COR_TEXTO_SECUNDARIO)
    estilizar_eixos(ax, grade="y")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Limiar de similaridade do agente de decisão", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("%", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    ax.set_title(f"Calibração do limiar de recusa (texto: {fonte})", loc="left", fontsize=10, color=COR_TEXTO,
                 pad=26)
    legenda = ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    for texto in legenda.get_texts():
        texto.set_color(COR_TEXTO)
    fig.tight_layout()
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


def avaliar_sistema(perguntas: pd.DataFrame, textos: dict[int, str], limiar: float, metodo: str) -> pd.DataFrame:
    """Roda o assistente completo (PLN + decisão) e classifica cada resultado."""
    assistente = AssistenteAcademico(agente=AgenteDecisao(limiar_similaridade=limiar), metodo_busca=metodo)
    linhas = []
    for p in perguntas.itertuples():
        if p.pagina not in textos:
            continue
        assistente.carregar_texto(textos[p.pagina])
        r = assistente.perguntar(p.pergunta)
        if p.intencao == "resumo":
            resultado = "resumo gerado" if r.acao == "resumir" else "errada"
        elif p.tem_resposta == 0:
            resultado = "recusa correta" if r.acao == "nao_encontrado" else "errada (inventou)"
        elif r.acao == "nao_encontrado":
            resultado = "recusa indevida"
        else:
            resultado = "correta" if contem_resposta(r.texto, p.resposta_esperada) else "errada"
        linhas.append({"pagina": p.pagina, "pergunta": p.pergunta, "intencao_real": p.intencao,
                       "intencao_prevista": r.intencao, "acao": r.acao, "regra": r.regra,
                       "resposta": r.texto, "resposta_esperada": p.resposta_esperada,
                       "resultado_automatico": resultado, "avaliacao_manual": ""})
    return pd.DataFrame(linhas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", choices=["gabarito", "ocr"], default="gabarito")
    args = parser.parse_args()

    perguntas = pd.read_csv(DADOS / "perguntas.csv", encoding="utf-8", keep_default_na=False)
    textos = carregar_textos(args.fonte)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)

    avaliar_intencao(perguntas).to_csv(RESULTADOS / "pln_intencao.csv", index=False, encoding="utf-8")
    busca, detalhes = avaliar_busca(perguntas, textos)
    busca.to_csv(RESULTADOS / f"pln_busca_{args.fonte}.csv", index=False, float_format="%.4f")
    detalhes.to_csv(RESULTADOS / f"pln_busca_detalhes_{args.fonte}.csv", index=False, encoding="utf-8")

    melhor_metodo = busca.sort_values(["mrr", "recall_1"], ascending=False)["metodo"].iloc[0]
    tabela, limiar = calibrar_limiar(perguntas, textos, melhor_metodo)
    tabela.to_csv(RESULTADOS / f"decisao_limiar_{args.fonte}.csv", index=False, float_format="%.4f")
    grafico_limiar(tabela, limiar, args.fonte, GRAFICOS / f"decisao_limiar_{args.fonte}.png")
    melhor = tabela[tabela["limiar"] == limiar].iloc[0]
    print(f"Decisão ({melhor_metodo}): limiar {limiar:.2f} -> acerto total {melhor['acerto_total']:.3f} | "
          f"respondíveis corretas {melhor['respondiveis_corretas']:.3f} | "
          f"sem resposta recusadas {melhor['sem_resposta_recusadas']:.3f}\n")

    sistema = avaliar_sistema(perguntas, textos, limiar, melhor_metodo)
    sistema.to_csv(RESULTADOS / f"respostas_{args.fonte}.csv", index=False, encoding="utf-8")
    print("Sistema completo:")
    print(sistema["resultado_automatico"].value_counts().to_string())


if __name__ == "__main__":
    main()
