import unicodedata

import cv2
import numpy as np
import pytest

from src.visao_computacional.avaliacao import cer, distancia_edicao, wer
from src.visao_computacional.gerar_imagens_sinteticas import quebrar_linhas, carregar_fonte, renderizar_texto
from src.visao_computacional.io_imagem import carregar_imagem, salvar_imagem
from src.visao_computacional.ocr import SegmentoOCR, easyocr_disponivel, montar_texto
from src.visao_computacional.preprocessamento import (
    ConfigPreprocessamento,
    corrigir_iluminacao,
    estimar_inclinacao,
    preprocessar,
    rotacionar,
)

TEXTO = (
    "A fotossíntese é o processo pelo qual plantas, algas e algumas bactérias convertem "
    "energia luminosa em energia química. Nos cloroplastos, a clorofila absorve a luz."
)


@pytest.fixture(scope="module")
def imagem_texto() -> np.ndarray:
    imagem, _ = renderizar_texto(TEXTO, tamanho_fonte=28)
    return np.array(imagem)


# --- Métricas -------------------------------------------------------------

def test_distancia_edicao_casos_conhecidos():
    assert distancia_edicao("gato", "gato") == 0
    assert distancia_edicao("gato", "pato") == 1
    assert distancia_edicao("", "abc") == 3
    assert distancia_edicao("kitten", "sitting") == 3


def test_cer_e_wer():
    assert cer("abcd", "abcf") == pytest.approx(0.25)
    assert wer("o gato subiu", "o pato subiu") == pytest.approx(1 / 3)


def test_quebra_de_linha_e_espacos_nao_contam_como_erro():
    assert cer("texto  com\nquebra", "texto com quebra") == 0


def test_acentos_em_formas_unicode_diferentes_sao_iguais():
    composto = "ação"
    decomposto = unicodedata.normalize("NFD", composto)
    assert composto != decomposto
    assert cer(composto, decomposto) == 0


# --- Pré-processamento ----------------------------------------------------

@pytest.mark.parametrize("angulo", [-6, -2.5, 3, 5.5])
def test_estimar_inclinacao_recupera_angulo(imagem_texto, angulo):
    inclinada = rotacionar(imagem_texto, angulo, cor_fundo=255)
    assert estimar_inclinacao(inclinada) == pytest.approx(-angulo, abs=0.3)


def test_estimar_inclinacao_em_imagem_vazia_devolve_zero():
    assert estimar_inclinacao(np.full((200, 300), 255, dtype=np.uint8)) == 0.0


def test_corrigir_iluminacao_remove_sombra(imagem_texto):
    # Escurece o lado esquerdo da imagem até 40% do brilho original.
    gradiente = np.linspace(0.4, 1.0, imagem_texto.shape[1])[np.newaxis, :]
    sombreada = (imagem_texto * gradiente).astype(np.uint8)
    corrigida = corrigir_iluminacao(sombreada)
    papel = imagem_texto == 255
    # Antes: o papel varia muito entre os lados. Depois: fica quase uniforme.
    assert sombreada[papel].std() > 30
    assert corrigida[papel].std() < 10


def test_preprocessar_binarizado_gera_so_preto_e_branco(imagem_texto):
    colorida = cv2.cvtColor(imagem_texto, cv2.COLOR_GRAY2BGR)
    resultado = preprocessar(colorida, ConfigPreprocessamento(binarizar=True))
    assert set(np.unique(resultado.imagem)) <= {0, 255}
    assert {"original", "cinza", "iluminacao_corrigida", "binarizada"} <= set(resultado.etapas)


# --- Entrada e saída ------------------------------------------------------

def test_salvar_e_carregar_em_caminho_com_acentos(tmp_path, imagem_texto):
    caminho = tmp_path / "Área de Trabalho" / "ação.png"
    salvar_imagem(imagem_texto, caminho)
    lida = carregar_imagem(caminho)
    assert lida.shape[:2] == imagem_texto.shape[:2]


# --- Imagens sintéticas e OCR ---------------------------------------------

def test_quebrar_linhas_respeita_largura():
    fonte = carregar_fonte("DejaVuSans.ttf", 24)
    linhas = quebrar_linhas(TEXTO, fonte, 400)
    assert len(linhas) > 1
    assert all(fonte.getlength(linha) <= 400 for linha in linhas)
    assert " ".join(linhas) == TEXTO


def test_montar_texto_ordena_em_linhas():
    segmentos = [
        SegmentoOCR("mundo", 0.9, (120, 12, 80, 20)),
        SegmentoOCR("segunda", 0.9, (10, 50, 90, 20)),
        SegmentoOCR("olá", 0.9, (10, 10, 60, 20)),
    ]
    assert montar_texto(segmentos) == "olá mundo\nsegunda"


@pytest.mark.lento
@pytest.mark.skipif(not easyocr_disponivel(), reason="EasyOCR não instalado")
def test_easyocr_le_texto_limpo(imagem_texto):
    from src.visao_computacional.ocr import criar_motor, ler_imagem

    resultado, _ = ler_imagem(cv2.cvtColor(imagem_texto, cv2.COLOR_GRAY2BGR), criar_motor("easyocr"))
    assert cer(TEXTO, resultado.texto) < 0.05
