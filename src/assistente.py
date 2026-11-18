"""Integração dos módulos num único fluxo.

    imagem -> [Visão Computacional] texto + confiança
           -> [Machine Learning]    matéria do material
    pergunta -> [PLN]               intenção, palavras-chave, trechos relevantes
             -> [Decisão]           escolhe a ação
             -> resposta + trecho de origem
"""

from dataclasses import dataclass, field

import numpy as np

from src.decisao.agente import AgenteDecisao, Percepcao
from src.machine_learning.classificador import ClassificadorMateria
from src.pln.busca import IndiceBusca, ResultadoBusca
from src.pln.intencao import palavras_chave, reconhecer_intencao
from src.pln.resposta import resposta_direta, resumo_extrativo
from src.visao_computacional.ocr import MotorOCR, criar_motor, ler_imagem
from src.visao_computacional.preprocessamento import ResultadoPreprocessamento

MENSAGENS = {
    "pedir_nova_foto": "Não consegui ler o material com segurança. Tente outra foto, com mais luz e o texto reto.",
    "nao_encontrado": "Não encontrei essa informação no material enviado.",
}


@dataclass
class Material:
    texto: str
    confianca_ocr: float
    materia: str | None
    probabilidade_materia: float
    probabilidades: dict[str, float]
    indice: IndiceBusca
    preprocessamento: ResultadoPreprocessamento | None = None


@dataclass
class Resposta:
    texto: str
    acao: str
    regra: str
    intencao: str
    palavras_chave: list[str]
    trechos: list[ResultadoBusca] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


class AssistenteAcademico:
    def __init__(
        self,
        motor_ocr: MotorOCR | None = None,
        classificador: ClassificadorMateria | None = None,
        agente: AgenteDecisao | None = None,
        metodo_busca: str = "caracteres",
    ):
        self._motor_ocr = motor_ocr
        self.classificador = classificador or ClassificadorMateria()
        self.agente = agente or AgenteDecisao()
        self.metodo_busca = metodo_busca
        self.material: Material | None = None

    @property
    def motor_ocr(self) -> MotorOCR:
        if self._motor_ocr is None:  # só carrega o OCR (lento) se for usar imagem
            self._motor_ocr = criar_motor("easyocr")
        return self._motor_ocr

    def carregar_imagem(self, imagem: np.ndarray) -> Material:
        resultado, preparado = ler_imagem(imagem, self.motor_ocr)
        return self._carregar(resultado.texto, resultado.confianca_media, preparado)

    def carregar_texto(self, texto: str, confianca_ocr: float = 1.0) -> Material:
        """Usa um texto já extraído (útil para testes e para os experimentos de PLN)."""
        return self._carregar(texto, confianca_ocr, None)

    def _carregar(self, texto: str, confianca: float, preparado) -> Material:
        materia, probabilidade, probabilidades = (None, 0.0, {})
        if texto.strip():
            materia, probabilidade, probabilidades = self.classificador.prever(texto)
        self.material = Material(texto, confianca, materia, probabilidade, probabilidades,
                                 IndiceBusca(texto, self.metodo_busca), preparado)
        return self.material

    def perguntar(self, pergunta: str) -> Resposta:
        if self.material is None:
            raise RuntimeError("Carregue uma imagem antes de perguntar.")
        material = self.material
        intencao = reconhecer_intencao(pergunta)
        chaves = palavras_chave(pergunta)
        trechos = material.indice.buscar(pergunta, k=3, intencao=intencao.tipo)

        materia_pergunta, prob_pergunta = None, 0.0
        if chaves:
            materia_pergunta, prob_pergunta, _ = self.classificador.prever(" ".join(chaves))

        decisao = self.agente.decidir(Percepcao(
            texto_ocr=material.texto,
            confianca_ocr=material.confianca_ocr,
            materia=material.materia,
            probabilidade_materia=material.probabilidade_materia,
            intencao=intencao.tipo,
            melhor_similaridade=trechos[0].similaridade if trechos else 0.0,
            materia_pergunta=materia_pergunta,
            probabilidade_materia_pergunta=prob_pergunta,
        ))

        if decisao.acao == "responder":
            texto = resposta_direta(trechos)
        elif decisao.acao == "resumir":
            frases = resumo_extrativo(material.indice, pergunta)
            texto = " ".join(f.texto for f in frases)
            trechos = [ResultadoBusca(f, 1.0) for f in frases]
        else:
            texto = MENSAGENS[decisao.acao]
            trechos = trechos if decisao.acao == "nao_encontrado" else []

        return Resposta(texto, decisao.acao, decisao.regra, intencao.tipo, chaves, trechos, decisao.avisos)
