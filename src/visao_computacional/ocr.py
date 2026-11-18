"""Reconhecimento óptico de caracteres (OCR).

Dois motores com a mesma interface, para poderem ser comparados nos experimentos:

- EasyOCR (padrão): rede neural de detecção (CRAFT) + reconhecimento (CRNN),
  pré-treinada. Instala só com pip e roda em CPU.
- Tesseract: motor clássico (LSTM) mantido pela comunidade. Mais leve, mas
  exige instalar o programa no sistema e o pacote do idioma português.

Uso pela linha de comando:
    python -m src.visao_computacional.ocr caminho/da/imagem.jpg [--motor tesseract] [--sem-preprocessamento]
"""

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .io_imagem import carregar_imagem
from .preprocessamento import ConfigPreprocessamento, ResultadoPreprocessamento, limitar_tamanho, preprocessar

TESSERACT_WINDOWS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass
class SegmentoOCR:
    """Um pedaço de texto detectado (palavra ou grupo de palavras) e onde ele está."""

    texto: str
    confianca: float  # de 0 a 1
    caixa: tuple[int, int, int, int]  # x, y, largura, altura


@dataclass
class ResultadoOCR:
    texto: str
    segmentos: list[SegmentoOCR]
    motor: str

    @property
    def confianca_media(self) -> float:
        """Média das confianças ponderada pelo tamanho de cada segmento.

        Ponderar evita que um "a" isolado com confiança baixa pese tanto quanto
        uma frase inteira lida com segurança.
        """
        pesos = [len(s.texto) for s in self.segmentos]
        if sum(pesos) == 0:
            return 0.0
        return float(np.average([s.confianca for s in self.segmentos], weights=pesos))


class MotorOCR(Protocol):
    nome: str

    def reconhecer(self, imagem: np.ndarray) -> ResultadoOCR: ...


def montar_texto(segmentos: list[SegmentoOCR]) -> str:
    """Coloca os segmentos em ordem de leitura.

    Os detectores não devolvem o texto na ordem certa. Agrupamos os segmentos em
    linhas (centros verticais próximos) e ordenamos cada linha da esquerda para
    a direita.
    """
    if not segmentos:
        return ""
    ordenados = sorted(segmentos, key=lambda s: s.caixa[1] + s.caixa[3] / 2)
    linhas: list[list[SegmentoOCR]] = []
    for segmento in ordenados:
        centro = segmento.caixa[1] + segmento.caixa[3] / 2
        if linhas:
            linha = linhas[-1]
            centro_linha = np.mean([s.caixa[1] + s.caixa[3] / 2 for s in linha])
            altura_linha = np.median([s.caixa[3] for s in linha])
            if abs(centro - centro_linha) < altura_linha / 2:
                linha.append(segmento)
                continue
        linhas.append([segmento])
    return "\n".join(
        " ".join(s.texto for s in sorted(linha, key=lambda s: s.caixa[0])) for linha in linhas
    )


class MotorEasyOCR:
    nome = "easyocr"

    def __init__(self, idiomas: tuple[str, ...] = ("pt",), gpu: bool = False):
        self.idiomas = list(idiomas)
        self.gpu = gpu
        self._leitor = None

    def _carregar(self):
        # O modelo é carregado só no primeiro uso: leva alguns segundos e, na
        # primeira execução, baixa os pesos da rede (~100 MB) para ~/.EasyOCR.
        if self._leitor is None:
            import easyocr

            with warnings.catch_warnings():
                # Avisos internos do PyTorch usado pelo EasyOCR; não afetam o resultado.
                warnings.simplefilter("ignore")
                self._leitor = easyocr.Reader(self.idiomas, gpu=self.gpu, verbose=False)
        return self._leitor

    def reconhecer(self, imagem: np.ndarray) -> ResultadoOCR:
        leitor = self._carregar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deteccoes = leitor.readtext(imagem, detail=1, paragraph=False)
        segmentos = []
        for pontos, texto, confianca in deteccoes:
            xs = [p[0] for p in pontos]
            ys = [p[1] for p in pontos]
            x, y = int(min(xs)), int(min(ys))
            caixa = (x, y, int(max(xs)) - x, int(max(ys)) - y)
            segmentos.append(SegmentoOCR(texto, float(confianca), caixa))
        return ResultadoOCR(montar_texto(segmentos), segmentos, self.nome)


class MotorTesseract:
    nome = "tesseract"

    def __init__(self, idioma: str = "por", psm: int = 3, executavel: str | Path | None = None):
        import pytesseract

        if executavel:
            pytesseract.pytesseract.tesseract_cmd = str(executavel)
        elif shutil.which("tesseract") is None and TESSERACT_WINDOWS.exists():
            pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_WINDOWS)
        self._pytesseract = pytesseract
        self.idioma = idioma
        # psm 3 = segmentação automática da página (o padrão do Tesseract).
        self.psm = psm

    def reconhecer(self, imagem: np.ndarray) -> ResultadoOCR:
        dados = self._pytesseract.image_to_data(
            imagem, lang=self.idioma, config=f"--psm {self.psm}",
            output_type=self._pytesseract.Output.DICT,
        )
        segmentos = []
        for i, texto in enumerate(dados["text"]):
            confianca = float(dados["conf"][i])
            # O Tesseract usa confiança -1 para blocos e linhas, que não são palavras.
            if confianca < 0 or not texto.strip():
                continue
            caixa = (dados["left"][i], dados["top"][i], dados["width"][i], dados["height"][i])
            segmentos.append(SegmentoOCR(texto, confianca / 100, caixa))
        return ResultadoOCR(montar_texto(segmentos), segmentos, self.nome)


def tesseract_disponivel() -> bool:
    try:
        import pytesseract
    except ImportError:
        return False
    return shutil.which("tesseract") is not None or TESSERACT_WINDOWS.exists()


def easyocr_disponivel() -> bool:
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return False
    return True


def criar_motor(nome: str = "easyocr", **opcoes) -> MotorOCR:
    if nome == "easyocr":
        if not easyocr_disponivel():
            raise RuntimeError("EasyOCR não está instalado. Rode: pip install -r requirements.txt")
        return MotorEasyOCR(**opcoes)
    if nome == "tesseract":
        if not tesseract_disponivel():
            raise RuntimeError(
                "Tesseract não encontrado. Instale o programa (com o idioma português) "
                "e veja as instruções no README."
            )
        return MotorTesseract(**opcoes)
    raise ValueError(f"Motor de OCR desconhecido: {nome}")


def ler_imagem(
    imagem: np.ndarray,
    motor: MotorOCR,
    config: ConfigPreprocessamento | None = None,
    usar_preprocessamento: bool = True,
) -> tuple[ResultadoOCR, ResultadoPreprocessamento | None]:
    """Pré-processa (opcional) e roda o OCR. Ponto de entrada usado pelo resto do sistema."""
    imagem = limitar_tamanho(imagem)
    if not usar_preprocessamento:
        return motor.reconhecer(imagem), None
    preparado = preprocessar(imagem, config)
    return motor.reconhecer(preparado.imagem), preparado


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai o texto de uma imagem.")
    parser.add_argument("imagem", type=Path)
    parser.add_argument("--motor", choices=["easyocr", "tesseract"], default="easyocr")
    parser.add_argument("--sem-preprocessamento", action="store_true")
    args = parser.parse_args()

    motor = criar_motor(args.motor)
    resultado, preparado = ler_imagem(
        carregar_imagem(args.imagem), motor, usar_preprocessamento=not args.sem_preprocessamento
    )
    sys.stdout.reconfigure(encoding="utf-8")
    print(resultado.texto)
    print(f"\n[motor: {resultado.motor} | confiança média: {resultado.confianca_media:.2f}", end="")
    if preparado is not None:
        print(f" | inclinação corrigida: {preparado.angulo_corrigido:.2f}°", end="")
    print("]")


if __name__ == "__main__":
    main()
