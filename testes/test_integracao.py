"""Teste de ponta a ponta: imagem -> OCR -> matéria -> pergunta -> decisão -> resposta."""

import numpy as np
import pytest

from src.machine_learning.classificador import CAMINHO_MODELO
from src.visao_computacional.ocr import easyocr_disponivel

pytestmark = [
    pytest.mark.lento,
    pytest.mark.skipif(not easyocr_disponivel(), reason="EasyOCR não instalado"),
    pytest.mark.skipif(not CAMINHO_MODELO.exists(), reason="Modelo de matéria não treinado"),
]


@pytest.fixture(scope="module")
def assistente():
    import random

    from src.assistente import AssistenteAcademico
    from src.visao_computacional.gerar_imagens_sinteticas import (
        ParametrosImagem, aplicar_degradacao, carregar_textos)
    from src.visao_computacional.gerar_material_impressao import desenhar_pagina

    itens = [t for t in carregar_textos() if t["materia"] == "Biologia"]
    pagina, _ = desenhar_pagina(1, "Biologia", itens, "DejaVuSerif.ttf")
    # Simula uma foto: inclinada, com sombra e ruído.
    foto = aplicar_degradacao(pagina, ParametrosImagem("DejaVuSerif.ttf", 29, angulo=4.0, desfoque=0.6,
                                                       ruido=8, sombra=0.4, cor_papel=225, cor_tinta=30,
                                                       qualidade_jpeg=80), np.random.default_rng(0))
    imagem = np.dstack([np.asarray(foto)] * 3)
    random.seed(0)
    a = AssistenteAcademico()
    a.carregar_imagem(imagem)
    return a


def test_material_identificado(assistente):
    assert assistente.material.materia == "Biologia"
    assert assistente.material.confianca_ocr > 0.5


def test_responde_pergunta_do_material(assistente):
    r = assistente.perguntar("Quais são as fases da fotossíntese?")
    assert r.acao == "responder"
    assert "fase clara" in r.texto.lower()


def test_recusa_pergunta_fora_do_material(assistente):
    assert assistente.perguntar("Quem foi o primeiro presidente do Brasil?").acao == "nao_encontrado"


def test_resume(assistente):
    r = assistente.perguntar("Resuma o texto")
    assert r.acao == "resumir" and len(r.trechos) >= 2
