"""Busca dos trechos do material mais relevantes para uma pergunta.

O texto extraído da imagem é dividido em frases. Cada frase (junto com o título
da sua seção) e a pergunta viram vetores TF-IDF, e a relevância é a
similaridade do cosseno entre eles: 1 quando apontam na mesma direção, 0 quando
não compartilham nenhum termo.

Dois métodos são comparados nos experimentos:
- "palavras":   termos inteiros. Simples, mas "fase" e "fases" viram termos diferentes.
- "caracteres": pedaços de 3 a 5 letras. Trata plural, variações da palavra e
                erros pequenos do OCR, porque as palavras ainda compartilham pedaços.

Duas heurísticas ajustam a pontuação final de cada frase:
- Contexto (anáfora): frases que começam com pronome ("Ela é dividida em
  prófase...", "Seu gráfico é...") falam do assunto da frase anterior. Por isso
  elas herdam metade da similaridade da frase anterior da mesma seção.
- Lista: em perguntas do tipo "quais são", frases com enumeração ("X, Y e Z")
  recebem o dobro de pontuação.
"""

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .intencao import palavras_chave
from .texto import Frase, dividir_em_frases, normalizar, palavras

METODOS = ("caracteres", "palavras")
PESO_CONTEXTO = 0.5
BONUS_LISTA = 2.0


@dataclass
class ResultadoBusca:
    frase: Frase
    similaridade: float  # pontuação de relevância (similaridade do cosseno + heurísticas)


PRONOMES_INICIAIS = re.compile(r"^(ela|ele|elas|eles|esse|essa|esses|essas|este|esta|estes|estas|"
                               r"isso|isto|seu|sua|seus|suas)\b")


def comeca_com_pronome(frase: str) -> bool:
    return bool(PRONOMES_INICIAIS.match(normalizar(frase)))


def tem_enumeracao(frase: str) -> bool:
    """Frase com pelo menos três itens separados por vírgula e/ou "e" (ex.: "A, B e C")."""
    return len(re.findall(r",", frase)) >= 2 or bool(re.search(r"\w+, [\w ]+ e \w+", frase))


def _preparar(texto: str) -> str:
    return " ".join(palavras(texto))


class IndiceBusca:
    def __init__(self, texto: str, metodo: str = "caracteres"):
        if metodo not in METODOS:
            raise ValueError(f"Método de busca desconhecido: {metodo}")
        self.metodo = metodo
        self.frases = dividir_em_frases(texto)
        self._vetorizador = None
        self._matriz = None
        if self.frases:
            documentos = [_preparar(f"{f.secao} {f.texto}") for f in self.frases]
            if metodo == "caracteres":
                self._vetorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            else:
                self._vetorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self._matriz = self._vetorizador.fit_transform(documentos)

    def buscar(self, pergunta: str, k: int = 3, intencao: str | None = None) -> list[ResultadoBusca]:
        termos = palavras_chave(pergunta)
        if self._vetorizador is None or not termos:
            return []
        consulta = self._vetorizador.transform([" ".join(termos)])
        cosseno = cosine_similarity(consulta, self._matriz)[0]

        pontuacao = cosseno.copy()
        for i in range(1, len(self.frases)):
            if self.frases[i].secao == self.frases[i - 1].secao and comeca_com_pronome(self.frases[i].texto):
                pontuacao[i] += PESO_CONTEXTO * cosseno[i - 1]
        if intencao == "lista":
            pontuacao *= np.array([BONUS_LISTA if tem_enumeracao(f.texto) else 1.0 for f in self.frases])

        ordem = pontuacao.argsort(kind="stable")[::-1][:k]
        return [ResultadoBusca(self.frases[i], float(pontuacao[i])) for i in ordem]

    def similaridade_com_todas(self) -> list[float]:
        """Similaridade de cada frase com o "centro" do texto (média de todas), usada no resumo."""
        if self._matriz is None:
            return []
        centro = np.asarray(self._matriz.mean(axis=0))
        return cosine_similarity(self._matriz, centro)[:, 0].tolist()
