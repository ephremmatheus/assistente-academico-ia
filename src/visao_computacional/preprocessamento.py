"""Pré-processamento de imagens para OCR.

Fotos de material didático costumam ter sombra, ruído, baixa resolução e papel
inclinado. Cada função abaixo trata um desses problemas. A função `preprocessar`
aplica a sequência configurada e devolve também as etapas intermediárias, que
servem para mostrar o "antes e depois" na interface e no artigo.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


def para_cinza(imagem: np.ndarray) -> np.ndarray:
    """Converte para tons de cinza. Para ler texto, a cor não ajuda e só aumenta o custo."""
    if imagem.ndim == 2:
        return imagem
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def redimensionar(cinza: np.ndarray, largura_minima: int = 1200) -> np.ndarray:
    """Amplia imagens pequenas. Letras com poucos pixels de altura são lidas com mais erro."""
    largura = cinza.shape[1]
    if largura >= largura_minima:
        return cinza
    fator = largura_minima / largura
    return cv2.resize(cinza, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)


def remover_ruido(cinza: np.ndarray, forca: int = 10) -> np.ndarray:
    """Suaviza o ruído (granulado da câmera) preservando as bordas das letras.

    O Non-Local Means compara pequenos blocos da imagem e faz a média entre
    blocos parecidos, em vez de borrar tudo por igual como um filtro gaussiano.
    """
    return cv2.fastNlMeansDenoising(cinza, None, h=forca, templateWindowSize=7, searchWindowSize=21)


def estimar_fundo(cinza: np.ndarray) -> np.ndarray:
    """Estima como seria o papel sem o texto (incluindo sombras e variações de luz).

    O fechamento morfológico (dilatação seguida de erosão) com um elemento maior
    que a espessura das letras "apaga" os traços escuros e deixa só o fundo.
    O cálculo é feito numa versão reduzida da imagem para ficar rápido.
    """
    altura, largura = cinza.shape[:2]
    escala = min(1.0, 600 / min(altura, largura))
    pequena = cv2.resize(cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else cinza
    tamanho = max(15, min(pequena.shape[:2]) // 20)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))
    fundo = cv2.morphologyEx(pequena, cv2.MORPH_CLOSE, elemento)
    fundo = cv2.GaussianBlur(fundo, (0, 0), tamanho / 2)
    return cv2.resize(fundo, (largura, altura), interpolation=cv2.INTER_LINEAR)


def corrigir_iluminacao(cinza: np.ndarray) -> np.ndarray:
    """Remove sombras dividindo a imagem pelo fundo estimado.

    Onde o papel está escuro por causa de uma sombra, o fundo também está escuro,
    e a divisão compensa. O resultado é papel quase branco por igual.
    """
    fundo = estimar_fundo(cinza)
    return cv2.divide(cinza, fundo, scale=255)


def estimar_inclinacao(cinza: np.ndarray, angulo_maximo: float = 10.0) -> float:
    """Estima o ângulo (em graus) que deixa as linhas de texto na horizontal.

    Heurística do perfil de projeção: some quantos pixels de tinta há em cada
    linha da imagem. Com o texto alinhado, a tinta se concentra em faixas
    (linhas de texto) separadas por faixas vazias, e a soma dos quadrados desse
    perfil é alta. Com o texto torto, as faixas se misturam e o valor cai.
    A busca é feita em duas fases: grossa (passo de 0,5°) e fina (0,05°) em
    torno do melhor ângulo encontrado.

    O retorno segue a convenção de `rotacionar`: aplicar
    `rotacionar(imagem, angulo)` com o valor devolvido endireita o texto.
    """
    escala = min(1.0, 800 / max(cinza.shape[:2]))
    pequena = cv2.resize(cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else cinza
    # Otsu escolhe sozinho o limiar entre tinta e papel; INV deixa a tinta com valor 255.
    _, tinta = cv2.threshold(pequena, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(tinta)
    if len(xs) < 50:
        return 0.0
    passo = max(1, len(xs) // 200_000)
    xs = xs[::passo].astype(np.float64)
    ys = ys[::passo].astype(np.float64)

    def energia_do_perfil(angulo: float) -> float:
        # Em vez de girar a imagem inteira, giramos só as coordenadas dos pixels de tinta.
        # A fórmula é a mesma da matriz de rotação usada pelo OpenCV em `rotacionar`.
        rad = np.deg2rad(angulo)
        y_girado = -np.sin(rad) * xs + np.cos(rad) * ys
        perfil = np.bincount(np.round(y_girado - y_girado.min()).astype(np.int64))
        return float(np.sum(perfil.astype(np.float64) ** 2))

    candidatos = np.arange(-angulo_maximo, angulo_maximo + 1e-9, 0.5)
    melhor = max(candidatos, key=energia_do_perfil)
    refinamento = np.arange(melhor - 0.5, melhor + 0.5 + 1e-9, 0.05)
    melhor = max(refinamento, key=energia_do_perfil)
    return round(float(melhor), 2)


def rotacionar(imagem: np.ndarray, angulo: float, cor_fundo: int = 255) -> np.ndarray:
    """Gira a imagem (ângulo positivo = sentido anti-horário) sem cortar os cantos."""
    altura, largura = imagem.shape[:2]
    centro = (largura / 2, altura / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos, sen = abs(matriz[0, 0]), abs(matriz[0, 1])
    nova_largura = int(round(altura * sen + largura * cos))
    nova_altura = int(round(altura * cos + largura * sen))
    matriz[0, 2] += nova_largura / 2 - centro[0]
    matriz[1, 2] += nova_altura / 2 - centro[1]
    borda = cor_fundo if imagem.ndim == 2 else (cor_fundo,) * imagem.shape[2]
    return cv2.warpAffine(
        imagem, matriz, (nova_largura, nova_altura),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=borda,
    )


def corrigir_inclinacao(cinza: np.ndarray, angulo_maximo: float = 10.0) -> tuple[np.ndarray, float]:
    """Endireita o texto. Devolve a imagem corrigida e o ângulo aplicado."""
    angulo = estimar_inclinacao(cinza, angulo_maximo)
    if abs(angulo) < 0.1:
        return cinza, 0.0
    return rotacionar(cinza, angulo, cor_fundo=255), angulo


def binarizar(cinza: np.ndarray, metodo: str = "otsu", tamanho_bloco: int = 31, constante: int = 15) -> np.ndarray:
    """Converte para preto e branco puro.

    - "otsu": um único limiar para a imagem toda, escolhido automaticamente.
      Funciona bem depois da correção de iluminação.
    - "adaptativo": cada região tem seu próprio limiar. Mais robusto quando a
      iluminação não foi corrigida, mas pode "esvaziar" letras muito grossas.
    """
    if metodo == "otsu":
        _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaria
    if metodo == "adaptativo":
        return cv2.adaptiveThreshold(
            cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, tamanho_bloco, constante
        )
    raise ValueError(f"Método de binarização desconhecido: {metodo}")


@dataclass
class ConfigPreprocessamento:
    """Liga e desliga cada etapa. Útil para medir o efeito de cada uma nos experimentos."""

    redimensionar: bool = True
    remover_ruido: bool = True
    corrigir_iluminacao: bool = True
    corrigir_inclinacao: bool = True
    binarizar: bool = False
    largura_minima: int = 1200
    metodo_binarizacao: str = "otsu"


@dataclass
class ResultadoPreprocessamento:
    imagem: np.ndarray
    etapas: dict[str, np.ndarray] = field(default_factory=dict)
    angulo_corrigido: float = 0.0


def preprocessar(imagem: np.ndarray, config: ConfigPreprocessamento | None = None) -> ResultadoPreprocessamento:
    """Aplica as etapas na ordem: cinza, tamanho, ruído, iluminação, inclinação, binarização.

    A ordem importa: o ruído é removido antes da correção de iluminação porque a
    divisão pelo fundo amplifica o ruído das áreas escuras; a inclinação é
    estimada depois da iluminação porque o limiar de Otsu funciona melhor com
    o fundo uniforme.
    """
    config = config or ConfigPreprocessamento()
    etapas = {"original": imagem}

    atual = para_cinza(imagem)
    etapas["cinza"] = atual

    if config.redimensionar:
        atual = redimensionar(atual, config.largura_minima)
    if config.remover_ruido:
        atual = remover_ruido(atual)
        etapas["sem_ruido"] = atual
    if config.corrigir_iluminacao:
        atual = corrigir_iluminacao(atual)
        etapas["iluminacao_corrigida"] = atual

    angulo = 0.0
    if config.corrigir_inclinacao:
        atual, angulo = corrigir_inclinacao(atual)
        etapas["inclinacao_corrigida"] = atual
    if config.binarizar:
        atual = binarizar(atual, config.metodo_binarizacao)
        etapas["binarizada"] = atual

    return ResultadoPreprocessamento(imagem=atual, etapas=etapas, angulo_corrigido=angulo)
