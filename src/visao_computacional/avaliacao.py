"""Métricas para avaliar o OCR.

CER (Character Error Rate) e WER (Word Error Rate) são as métricas padrão de
reconhecimento de texto. Ambas usam a distância de edição (Levenshtein): o
menor número de substituições (S), remoções (D) e inserções (I) para
transformar o texto reconhecido no texto correto.

    CER = (S + D + I) / número de caracteres do texto correto
    WER = (S + D + I) / número de palavras do texto correto

0 significa leitura perfeita. O valor pode passar de 1 se o OCR inventar
muito texto a mais.
"""

import re
import unicodedata
from collections.abc import Sequence


def normalizar_texto(texto: str, ignorar_maiusculas: bool = False) -> str:
    """Padroniza o texto antes de comparar.

    - NFC: "ç" pode ser gravado como um caractere só ou como "c" + cedilha.
      Sem normalizar, textos visualmente iguais seriam contados como diferentes.
    - Espaços: quebras de linha e espaços repetidos viram um espaço só, porque a
      quebra de linha depende da largura da imagem e não é erro de leitura.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.lower() if ignorar_maiusculas else texto


def distancia_edicao(a: Sequence, b: Sequence) -> int:
    """Distância de Levenshtein por programação dinâmica, guardando só duas linhas da tabela."""
    if len(a) < len(b):
        a, b = b, a
    anterior = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        atual = [i]
        for j, item_b in enumerate(b, start=1):
            atual.append(min(
                anterior[j] + 1,                      # remoção
                atual[j - 1] + 1,                     # inserção
                anterior[j - 1] + (item_a != item_b),  # substituição (custo 0 se iguais)
            ))
        anterior = atual
    return anterior[-1]


def cer(referencia: str, hipotese: str, ignorar_maiusculas: bool = False) -> float:
    ref = normalizar_texto(referencia, ignorar_maiusculas)
    hip = normalizar_texto(hipotese, ignorar_maiusculas)
    if not ref:
        return 0.0 if not hip else 1.0
    return distancia_edicao(ref, hip) / len(ref)


def wer(referencia: str, hipotese: str, ignorar_maiusculas: bool = False) -> float:
    ref = normalizar_texto(referencia, ignorar_maiusculas).split()
    hip = normalizar_texto(hipotese, ignorar_maiusculas).split()
    if not ref:
        return 0.0 if not hip else 1.0
    return distancia_edicao(ref, hip) / len(ref)
