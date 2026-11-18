"""Agente de decisão baseado em regras (agente reativo simples).

Ciclo do agente:
1. Percepção: reúne o que os outros módulos produziram (texto e confiança do
   OCR, matéria prevista, intenção da pergunta, trechos encontrados).
2. Decisão: aplica regras condição -> ação, em ordem de prioridade. A primeira
   regra cuja condição é verdadeira define a ação.
3. Ação: responder, resumir, pedir uma nova foto ou dizer que a informação não
   está no material.

A regra mais importante é a R4: se nenhum trecho é parecido o bastante com a
pergunta, o agente prefere dizer "não encontrei" a devolver um trecho
irrelevante. O limiar de similaridade foi calibrado no Experimento 4.
"""

from dataclasses import dataclass, field

LIMIAR_SIMILARIDADE_PADRAO = 0.30  # calibrado no Experimento 4 (resultados/decisao_limiar_gabarito.csv)


@dataclass
class Percepcao:
    texto_ocr: str
    confianca_ocr: float
    materia: str | None
    probabilidade_materia: float
    intencao: str
    melhor_similaridade: float
    materia_pergunta: str | None = None
    probabilidade_materia_pergunta: float = 0.0


@dataclass
class Decisao:
    acao: str           # "responder", "resumir", "pedir_nova_foto" ou "nao_encontrado"
    regra: str          # qual regra foi aplicada (para explicar a decisão)
    avisos: list[str] = field(default_factory=list)


class AgenteDecisao:
    def __init__(
        self,
        limiar_similaridade: float = LIMIAR_SIMILARIDADE_PADRAO,
        confianca_ocr_minima: float = 0.30,
        confianca_ocr_aviso: float = 0.60,
        probabilidade_materia_aviso: float = 0.60,
        minimo_caracteres: int = 20,
    ):
        self.limiar_similaridade = limiar_similaridade
        self.confianca_ocr_minima = confianca_ocr_minima
        self.confianca_ocr_aviso = confianca_ocr_aviso
        self.probabilidade_materia_aviso = probabilidade_materia_aviso
        self.minimo_caracteres = minimo_caracteres

    def decidir(self, p: Percepcao) -> Decisao:
        avisos = self._avisos(p)

        if len(p.texto_ocr.strip()) < self.minimo_caracteres:
            return Decisao("pedir_nova_foto", "R1: quase nenhum texto foi encontrado na imagem", avisos)
        if p.confianca_ocr < self.confianca_ocr_minima:
            return Decisao("pedir_nova_foto",
                           f"R2: confiança do OCR ({p.confianca_ocr:.0%}) abaixo de {self.confianca_ocr_minima:.0%}",
                           avisos)
        if p.intencao == "resumo":
            return Decisao("resumir", "R3: a pergunta pede um resumo", avisos)
        if p.melhor_similaridade < self.limiar_similaridade:
            return Decisao("nao_encontrado",
                           f"R4: melhor trecho com similaridade {p.melhor_similaridade:.2f}, "
                           f"abaixo do limiar {self.limiar_similaridade:.2f}", avisos)
        return Decisao("responder",
                       f"R5: trecho com similaridade {p.melhor_similaridade:.2f} (intenção: {p.intencao})", avisos)

    def _avisos(self, p: Percepcao) -> list[str]:
        avisos = []
        if self.confianca_ocr_minima <= p.confianca_ocr < self.confianca_ocr_aviso:
            avisos.append(f"A leitura da imagem teve confiança baixa ({p.confianca_ocr:.0%}). "
                          "Confira o trecho original.")
        if (p.materia and p.materia_pergunta and p.materia_pergunta != p.materia
                and p.probabilidade_materia_pergunta >= self.probabilidade_materia_aviso):
            avisos.append(f"Sua pergunta parece ser de {p.materia_pergunta}, "
                          f"mas o material enviado é de {p.materia}.")
        return avisos
