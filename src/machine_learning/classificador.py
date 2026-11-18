"""Classificador de matéria (aprendizado supervisionado com TF-IDF).

O TF-IDF transforma cada texto num vetor: cada posição é uma palavra (ou par de
palavras), com peso alto quando ela é frequente no texto e rara no conjunto
inteiro. Palavras como "célula" ou "ácido" ganham peso; palavras comuns a todas
as matérias quase não contam. Sobre esses vetores, um classificador linear
aprende quais termos indicam cada matéria.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.caminhos import DADOS, RAIZ
from src.pln.texto import STOPWORDS_SEM_ACENTO

DADOS_TREINO = DADOS / "wikipedia_materias.csv"
CAMINHO_MODELO = RAIZ / "modelos" / "classificador_materia.joblib"


def _vetorizador_palavras() -> TfidfVectorizer:
    return TfidfVectorizer(
        strip_accents="unicode", lowercase=True, stop_words=STOPWORDS_SEM_ACENTO,
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    )


def _vetorizador_caracteres() -> TfidfVectorizer:
    # Pedaços de 3 a 5 letras dentro das palavras. Tolera erros de OCR
    # ("fotossintese" e "fotossíntcse" ainda compartilham vários pedaços).
    return TfidfVectorizer(
        strip_accents="unicode", lowercase=True, analyzer="char_wb",
        ngram_range=(3, 5), min_df=2, sublinear_tf=True,
    )


MODELOS = {
    "Regressão Logística (palavras)": lambda: Pipeline([
        ("tfidf", _vetorizador_palavras()), ("clf", LogisticRegression(max_iter=3000, C=10))]),
    "Naive Bayes (palavras)": lambda: Pipeline([
        ("tfidf", _vetorizador_palavras()), ("clf", MultinomialNB(alpha=0.1))]),
    "SVM linear (palavras)": lambda: Pipeline([
        ("tfidf", _vetorizador_palavras()), ("clf", LinearSVC(C=1.0))]),
    "SVM linear (caracteres)": lambda: Pipeline([
        ("tfidf", _vetorizador_caracteres()), ("clf", LinearSVC(C=1.0))]),
}


def carregar_dados_treino(caminho: Path = DADOS_TREINO) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8")


def criar_modelo(nome: str, com_probabilidade: bool = False) -> Pipeline:
    """Cria o pipeline. Com `com_probabilidade`, o SVM é calibrado para devolver probabilidades."""
    modelo = MODELOS[nome]()
    if com_probabilidade and not hasattr(modelo.named_steps["clf"], "predict_proba"):
        modelo.steps[-1] = ("clf", CalibratedClassifierCV(modelo.named_steps["clf"], cv=3))
    return modelo


def salvar(modelo: Pipeline, nome: str, caminho: Path = CAMINHO_MODELO) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"nome": nome, "modelo": modelo}, caminho)


class ClassificadorMateria:
    """Interface usada pelo restante do sistema."""

    def __init__(self, caminho: Path = CAMINHO_MODELO):
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho}. Rode: python -m src.machine_learning.treinar"
            )
        salvo = joblib.load(caminho)
        self.nome = salvo["nome"]
        self.modelo = salvo["modelo"]

    def prever(self, texto: str) -> tuple[str, float, dict[str, float]]:
        """Devolve a matéria mais provável, a probabilidade dela e a de todas as matérias."""
        probabilidades = self.modelo.predict_proba([texto])[0]
        classes = self.modelo.classes_
        por_materia = dict(sorted(zip(classes, map(float, probabilidades)), key=lambda x: -x[1]))
        materia = next(iter(por_materia))
        return materia, por_materia[materia], por_materia
