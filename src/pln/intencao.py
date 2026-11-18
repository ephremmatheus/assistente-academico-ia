"""Reconhecimento de intenção da pergunta, por regras.

Cada intenção pede um tipo de resposta diferente:
- definicao: "O que é X?"         -> a frase que define X
- lista:     "Quais são as fases?" -> a frase que enumera os itens
- resumo:    "Resuma o texto"      -> as frases mais representativas do material
- factual:   qualquer outra pergunta (quando, onde, quem, como...) -> a frase mais relevante

As regras são expressões regulares aplicadas à pergunta sem acentos e em
minúsculas, testadas em ordem. A primeira que casar define a intenção.
Regras foram escolhidas em vez de um modelo treinado porque não há dados
rotulados de perguntas suficientes, e porque regras são fáceis de explicar e
de corrigir.
"""

import re
from dataclasses import dataclass

from .texto import normalizar, palavras

REGRAS = [
    ("resumo", r"\b(resum\w*|sintetiz\w*|principais (ideias|pontos)|do que (se )?trata|sobre o que (e|fala))\b"),
    ("definicao", r"\b(o que (e|sao|significa|quer dizer|afirma|diz|estabelece)|defin[aei]\w*|"
                  r"conceito de|significado de)\b"),
    ("lista", r"\b(quais (sao|foram|as|os)|liste|listar|cite|enumere|etapas|fases|tipos|exemplos)\b"),
]

PALAVRAS_DE_PERGUNTA = {
    "o", "que", "qual", "quais", "quando", "onde", "quem", "como", "porque", "quanto", "quanta",
    "quantos", "quantas", "explique", "defina", "definicao", "cite", "liste", "resuma", "resumo",
    "texto", "pagina", "material", "conteudo", "sao", "e", "faca", "diga", "fale",
}


@dataclass
class Intencao:
    tipo: str
    regra: str  # trecho da pergunta que ativou a regra (vazio quando caiu no padrão "factual")


def reconhecer_intencao(pergunta: str) -> Intencao:
    texto = normalizar(pergunta)
    for tipo, padrao in REGRAS:
        encontrado = re.search(padrao, texto)
        if encontrado:
            return Intencao(tipo, encontrado.group(0))
    return Intencao("factual", "")


def palavras_chave(pergunta: str) -> list[str]:
    """Palavras que carregam o assunto da pergunta (sem stopwords e sem palavras interrogativas)."""
    return [p for p in palavras(pergunta) if p not in PALAVRAS_DE_PERGUNTA and len(p) > 1]
