"""Funções básicas de texto usadas pelo PLN e pelo Machine Learning."""

import re
import unicodedata
from dataclasses import dataclass

# Palavras muito frequentes que não ajudam a identificar assunto nem a buscar trechos.
STOPWORDS = set("""
a à ao aos aquela aquelas aquele aqueles aquilo as às até com como da das de dela delas dele deles
depois do dos e é ela elas ele eles em entre era eram essa essas esse esses esta está estão estas
este estes eu foi foram há isso isto já lhe lhes mais mas me mesmo meu minha muito na nas nem no nos
nós num numa o os ou para pela pelas pelo pelos por qual quais quando que quem se sem ser seu seus
sua suas são também te tem têm um uma umas uns você vocês sobre cada onde ainda assim pode podem
sendo será seja sejam sua seus outro outra outros outras pois porque já só
""".split())


def sem_acentos(texto: str) -> str:
    decomposto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


STOPWORDS_SEM_ACENTO = sorted({sem_acentos(p) for p in STOPWORDS})


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e com espaços simples. Usado para comparar textos."""
    texto = sem_acentos(unicodedata.normalize("NFC", texto).lower())
    return re.sub(r"\s+", " ", texto).strip()


def palavras(texto: str, remover_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalizar(texto))
    if remover_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_SEM_ACENTO]
    return tokens


@dataclass
class Frase:
    texto: str      # a frase como aparece no material
    secao: str      # título da seção em que a frase está (pode ser vazio)
    posicao: int    # ordem da frase no texto


def _eh_titulo(linha: str, anterior_terminou_frase: bool) -> bool:
    """Títulos são linhas curtas, sem pontuação final, começando com maiúscula, que
    aparecem depois de uma frase terminada (ou no início). Uma linha que continua
    uma frase da linha anterior nunca é título."""
    return (anterior_terminou_frase and len(linha.split()) <= 8 and linha[:1].isupper()
            and not re.search(r"[.!?:;,]$", linha))


def dividir_em_frases(texto: str) -> list[Frase]:
    """Divide o texto (por exemplo, a saída do OCR) em frases, guardando a seção de cada uma.

    O OCR quebra o texto em linhas conforme a largura da imagem, então primeiro
    juntamos as linhas de um mesmo parágrafo e só depois separamos as frases
    pela pontuação.
    """
    frases: list[Frase] = []
    secao, paragrafo = "", []

    def fechar_paragrafo() -> None:
        if not paragrafo:
            return
        corrido = re.sub(r"-\s+(?=[a-zà-ú])", "", " ".join(paragrafo))  # desfaz hifenização de fim de linha
        for pedaco in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])", corrido):
            pedaco = pedaco.strip()
            if len(pedaco.split()) >= 3:
                frases.append(Frase(pedaco, secao, len(frases)))
        paragrafo.clear()

    anterior_terminou_frase = True
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            fechar_paragrafo()
            anterior_terminou_frase = True
        elif _eh_titulo(linha, anterior_terminou_frase):
            fechar_paragrafo()
            if not re.fullmatch(r"p[áa]gina \d+", linha.lower()):
                secao = linha
            anterior_terminou_frase = True
        else:
            paragrafo.append(linha)
            anterior_terminou_frase = bool(re.search(r"[.!?]$", linha))
    fechar_paragrafo()
    return frases
