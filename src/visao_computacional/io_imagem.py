"""Leitura e escrita de imagens.

No Windows, cv2.imread e cv2.imwrite falham quando o caminho tem acento
(ex.: "Área de Trabalho") e devolvem None sem erro. Para evitar isso, os bytes
do arquivo são lidos com NumPy e decodificados com cv2.imdecode.
"""

from pathlib import Path

import cv2
import numpy as np


def carregar_imagem(caminho: str | Path) -> np.ndarray:
    """Lê uma imagem do disco no formato BGR do OpenCV."""
    caminho = Path(caminho)
    dados = np.fromfile(caminho, dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    return imagem


def carregar_imagem_de_bytes(conteudo: bytes) -> np.ndarray:
    """Decodifica uma imagem recebida em memória (ex.: upload na interface)."""
    imagem = cv2.imdecode(np.frombuffer(conteudo, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError("O arquivo enviado não é uma imagem válida.")
    return imagem


def salvar_imagem(imagem: np.ndarray, caminho: str | Path) -> None:
    """Salva a imagem, criando as pastas necessárias."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".png", imagem)
    if not ok:
        raise ValueError(f"Não foi possível codificar a imagem para {caminho}")
    buffer.tofile(caminho)
