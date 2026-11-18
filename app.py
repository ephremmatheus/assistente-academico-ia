"""Interface do Assistente Acadêmico Inteligente.

Uso:
    streamlit run app.py
"""

import cv2
import streamlit as st

from src.assistente import AssistenteAcademico
from src.machine_learning.classificador import ClassificadorMateria
from src.visao_computacional.io_imagem import carregar_imagem_de_bytes
from src.visao_computacional.ocr import MotorEasyOCR

st.set_page_config(page_title="Assistente Acadêmico", page_icon="📚", layout="wide")

ROTULOS_ACAO = {
    "responder": "✅ Resposta encontrada no material",
    "resumir": "📝 Resumo do material",
    "nao_encontrado": "🔍 Informação não encontrada",
    "pedir_nova_foto": "📷 Foto ilegível",
}


@st.cache_resource(show_spinner="Carregando os modelos (só na primeira vez)...")
def carregar_modelos():
    return MotorEasyOCR(), ClassificadorMateria()


@st.cache_data(show_spinner=False)
def ler_material(conteudo: bytes):
    """Roda OCR + classificação uma única vez por imagem enviada."""
    motor, classificador = carregar_modelos()
    assistente = AssistenteAcademico(motor_ocr=motor, classificador=classificador)
    material = assistente.carregar_imagem(carregar_imagem_de_bytes(conteudo))
    preprocessada = material.preprocessamento.imagem if material.preprocessamento else None
    return material.texto, material.confianca_ocr, preprocessada


def assistente_com_texto(texto: str, confianca: float) -> AssistenteAcademico:
    motor, classificador = carregar_modelos()
    assistente = AssistenteAcademico(motor_ocr=motor, classificador=classificador)
    assistente.carregar_texto(texto, confianca)
    return assistente


st.title("📚 Assistente Acadêmico Inteligente")
st.caption("Envie a foto de um slide, apostila ou página de livro e faça perguntas sobre o conteúdo. "
           "As respostas são sempre trechos do próprio material.")

with st.sidebar:
    st.header("1. Material")
    arquivo = st.file_uploader("Foto do material", type=["jpg", "jpeg", "png"])
    st.divider()
    st.markdown("**Como funciona**")
    st.markdown(
        "1. **Visão Computacional**: corrige sombra e inclinação e lê o texto (OCR).\n"
        "2. **Machine Learning**: identifica a matéria.\n"
        "3. **PLN**: entende a pergunta e busca o trecho mais relevante.\n"
        "4. **Agente de decisão**: escolhe entre responder, resumir, pedir outra foto "
        "ou dizer que não encontrou."
    )

if arquivo is None:
    st.info("Envie uma imagem na barra lateral para começar.")
    st.stop()

conteudo = arquivo.getvalue()
if st.session_state.get("arquivo_atual") != arquivo.file_id:
    st.session_state["arquivo_atual"] = arquivo.file_id
    st.session_state["historico"] = []

with st.spinner("Lendo o material..."):
    texto, confianca, preprocessada = ler_material(conteudo)
assistente = assistente_com_texto(texto, confianca)
material = assistente.material

coluna_imagem, coluna_info = st.columns([1, 1])
with coluna_imagem:
    aba_original, aba_tratada = st.tabs(["Original", "Pré-processada"])
    aba_original.image(conteudo, width="stretch")
    if preprocessada is not None:
        aba_tratada.image(cv2.cvtColor(preprocessada, cv2.COLOR_GRAY2RGB) if preprocessada.ndim == 2
                          else preprocessada, width="stretch")
with coluna_info:
    m1, m2 = st.columns(2)
    m1.metric("Matéria", material.materia or "—", f"{material.probabilidade_materia:.0%} de probabilidade",
              delta_color="off")
    m2.metric("Confiança da leitura", f"{material.confianca_ocr:.0%}")
    with st.expander("Texto extraído", expanded=False):
        st.text(material.texto or "(nenhum texto encontrado)")

st.header("2. Pergunte")
for pergunta_antiga, resposta_antiga in st.session_state["historico"]:
    st.chat_message("user").write(pergunta_antiga)
    st.chat_message("assistant").markdown(resposta_antiga)

pergunta = st.chat_input("Ex.: O que é a fotossíntese?")
if pergunta:
    r = assistente.perguntar(pergunta)
    partes = [f"**{ROTULOS_ACAO[r.acao]}**", "", r.texto]
    if r.acao in ("responder", "resumir") and r.trechos:
        secao = r.trechos[0].frase.secao
        if secao:
            partes += ["", f"📍 Trecho da seção *{secao}*"]
    for aviso in r.avisos:
        partes += ["", f"⚠️ {aviso}"]
    partes += ["", f"<sub>Intenção: {r.intencao} · palavras-chave: {', '.join(r.palavras_chave) or '—'} · "
                   f"decisão: {r.regra}</sub>"]
    resposta_md = "\n".join(partes)
    st.chat_message("user").write(pergunta)
    st.chat_message("assistant").markdown(resposta_md, unsafe_allow_html=True)
    st.session_state["historico"].append((pergunta, resposta_md))
