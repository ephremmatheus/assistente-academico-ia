import pytest

from src.decisao.agente import AgenteDecisao, Percepcao
from src.pln.busca import IndiceBusca
from src.pln.intencao import palavras_chave, reconhecer_intencao
from src.pln.resposta import resumo_extrativo
from src.pln.texto import dividir_em_frases, normalizar

TEXTO_OCR = """Biologia
Fotossíntese
A fotossíntese é o processo pelo qual plantas, algas e algumas
bactérias convertem energia luminosa em energia química. Nos
cloroplastos, a clorofila absorve a luz. Esse processo ocorre em
duas fases: a fase clara e a fase escura, também chamada de ciclo de Calvin.
Mitose e meiose
A mitose é a divisão celular que origina duas células-filhas
idênticas. Ela é dividida em prófase, metáfase, anáfase e telófase.
Página 1"""


# --- Texto ----------------------------------------------------------------

def test_normalizar_remove_acentos_e_espacos():
    assert normalizar("  Fotossíntese   É  ") == "fotossintese e"


def test_dividir_em_frases_junta_linhas_e_guarda_secao():
    frases = dividir_em_frases(TEXTO_OCR)
    assert frases[0].texto.startswith("A fotossíntese é o processo")
    assert frases[0].texto.endswith("energia química.")
    assert frases[0].secao == "Fotossíntese"
    assert frases[-1].secao == "Mitose e meiose"
    assert all("Página" not in f.texto for f in frases)


# --- Intenção -------------------------------------------------------------

@pytest.mark.parametrize("pergunta, esperado", [
    ("O que é a fotossíntese?", "definicao"),
    ("Quais são as fases da mitose?", "lista"),
    ("Resuma o texto", "resumo"),
    ("Do que trata o material?", "resumo"),
    ("Onde ocorre a fotossíntese?", "factual"),
    ("Quando foi proclamada a independência?", "factual"),
])
def test_reconhecer_intencao(pergunta, esperado):
    assert reconhecer_intencao(pergunta).tipo == esperado


def test_palavras_chave_ignora_interrogativas_e_stopwords():
    assert palavras_chave("O que é a fotossíntese?") == ["fotossintese"]


# --- Busca e resposta -----------------------------------------------------

@pytest.mark.parametrize("metodo", ["caracteres", "palavras"])
def test_busca_encontra_frase_certa(metodo):
    # A resposta está numa frase que usa pronome ("Ela é dividida em prófase..."):
    # só é encontrada graças às heurísticas de contexto e de lista.
    indice = IndiceBusca(TEXTO_OCR, metodo)
    melhor = indice.buscar("Quais são as etapas da mitose?", intencao="lista")[0]
    assert "prófase" in melhor.frase.texto


def test_tem_enumeracao():
    from src.pln.busca import tem_enumeracao

    assert tem_enumeracao("Ela é dividida em prófase, metáfase, anáfase e telófase.")
    assert not tem_enumeracao("A mitose é a divisão celular.")


def test_busca_tolera_erro_de_ocr_com_caracteres():
    indice = IndiceBusca(TEXTO_OCR.replace("cloroplastos", "cl0roplastos"), "caracteres")
    assert "absorve a luz" in indice.buscar("O que acontece nos cloroplastos?")[0].frase.texto


def test_pergunta_sem_palavras_chave_nao_busca():
    assert IndiceBusca(TEXTO_OCR).buscar("O que é?") == []


def test_resumo_respeita_secao_citada_e_ordem():
    frases = resumo_extrativo(IndiceBusca(TEXTO_OCR), "Resuma o texto sobre mitose")
    assert frases and all(f.secao == "Mitose e meiose" for f in frases)
    assert [f.posicao for f in frases] == sorted(f.posicao for f in frases)


# --- Agente de decisão ----------------------------------------------------

def _percepcao(**mudancas) -> Percepcao:
    base = dict(texto_ocr=TEXTO_OCR, confianca_ocr=0.9, materia="Biologia", probabilidade_materia=0.9,
                intencao="factual", melhor_similaridade=0.5)
    base.update(mudancas)
    return Percepcao(**base)


@pytest.mark.parametrize("mudancas, acao", [
    (dict(texto_ocr="  "), "pedir_nova_foto"),
    (dict(confianca_ocr=0.1), "pedir_nova_foto"),
    (dict(intencao="resumo", melhor_similaridade=0.0), "resumir"),
    (dict(melhor_similaridade=0.05), "nao_encontrado"),
    (dict(), "responder"),
])
def test_regras_do_agente(mudancas, acao):
    assert AgenteDecisao(limiar_similaridade=0.2).decidir(_percepcao(**mudancas)).acao == acao


def test_agente_avisa_materia_diferente_e_confianca_baixa():
    decisao = AgenteDecisao().decidir(_percepcao(confianca_ocr=0.45, materia_pergunta="História",
                                                 probabilidade_materia_pergunta=0.8))
    assert decisao.acao == "responder"
    assert len(decisao.avisos) == 2
