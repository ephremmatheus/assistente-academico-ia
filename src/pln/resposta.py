"""Geração da resposta a partir dos trechos encontrados (abordagem extrativa).

A resposta é sempre um trecho do próprio material, nunca um texto gerado.
Assim o aluno vê exatamente de onde veio a informação e o sistema não inventa
conteúdo que não está na imagem.
"""

from .busca import IndiceBusca, ResultadoBusca
from .intencao import palavras_chave
from .texto import Frase


def resumo_extrativo(indice: IndiceBusca, pergunta: str = "", maximo_frases: int = 3) -> list[Frase]:
    """Escolhe as frases mais representativas, mantendo a ordem original do texto.

    Se a pergunta cita um assunto ("resuma o texto sobre mitose"), o resumo usa
    só a seção mais relacionada a ele. A representatividade de uma frase é a
    similaridade dela com o centro (a média) de todas as frases consideradas.
    """
    frases = indice.frases
    if not frases:
        return []
    if palavras_chave(pergunta):
        melhores = indice.buscar(pergunta, k=1)
        if melhores and melhores[0].similaridade > 0 and melhores[0].frase.secao:
            secao = melhores[0].frase.secao
            frases = [f for f in frases if f.secao == secao]
    pontuacao = dict(zip((f.posicao for f in indice.frases), indice.similaridade_com_todas()))
    escolhidas = sorted(frases, key=lambda f: -pontuacao[f.posicao])[:maximo_frases]
    return sorted(escolhidas, key=lambda f: f.posicao)


def resposta_direta(resultados: list[ResultadoBusca]) -> str:
    """Para definição, lista e pergunta factual: a frase mais relevante."""
    return resultados[0].frase.texto if resultados else ""
