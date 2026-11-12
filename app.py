"""Aplicação web (Streamlit) — Roubo × Furto de celulares em São Paulo (MLP).
Execute:  streamlit run app.py
Usa EXATAMENTE os atributos do treino: bairro, logradouro, período, subtipo do local, marca e dia da semana.
"""
import json, os
import numpy as np
import streamlit as st

# Pasta "artefatos" ao lado do app.py (funciona de qualquer diretório de execução)
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artefatos")
ARQUIVOS = ["modelo_mlp.keras", "colunas_treino.json", "categorias.json"]
st.set_page_config(page_title="MLP Roubo × Furto — SP", page_icon="📱", layout="centered")

@st.cache_resource
def carregar():
    from tensorflow import keras
    modelo = keras.models.load_model(os.path.join(ART, "modelo_mlp.keras"))
    with open(os.path.join(ART, "colunas_treino.json"), encoding="utf-8") as f:
        colunas = json.load(f)
    with open(os.path.join(ART, "categorias.json"), encoding="utf-8") as f:
        categorias = json.load(f)
    meta = {}
    p = os.path.join(ART, "metricas.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
    return modelo, colunas, categorias, meta

def vetor(entrada, colunas):
    """One-Hot idêntico ao do treino: 1 na coluna '<ATRIBUTO>_<categoria>'."""
    x = np.zeros((1, len(colunas)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(colunas)}
    for atributo, valor in entrada.items():
        col = f"{atributo}_{valor}"
        if col in idx:
            x[0, idx[col]] = 1.0
    return x

st.title("📱 Classificador de ocorrências: Roubo × Furto")
st.caption("Subtração de celulares — cidade de São Paulo (SSP-SP, 2025) · Rede neural MLP")

st.info(
    "**Como interpretar:** o modelo analisa as características de uma ocorrência **que já aconteceu (ou é hipotética)** "
    "e informa a probabilidade de ela ser *roubo* ou *furto*. Ele **não** calcula o risco de um crime ocorrer "
    "em determinado lugar ou horário."
)

faltando = [a for a in ARQUIVOS if not os.path.exists(os.path.join(ART, a))]
if faltando:
    st.error(
        f"Arquivos não encontrados em: {ART}\n\nFaltando: {', '.join(faltando)}.\n\n"
        "Rode o notebook inteiro (Run All) na mesma pasta do app.py, ou copie a pasta 'artefatos' gerada por ele "
        "(se rodou no Colab, baixe-a de lá) para ao lado do app.py."
    )
    st.stop()
try:
    modelo, colunas, categorias, meta = carregar()
except Exception as e:
    st.error(f"Os arquivos existem, mas o modelo não carregou. Detalhe: {e}")
    st.stop()

st.subheader("Características da ocorrência")
c1, c2 = st.columns(2)
with c1:
    bairro = st.selectbox("Bairro", categorias["BAIRRO"], help="'OUTROS' agrupa bairros raros na base de treino.")
    logradouro = st.selectbox("Logradouro", categorias["LOGRADOURO"], help="Os 300 logradouros mais frequentes no treino; 'OUTROS' para os demais.")
    sublocal = st.selectbox("Subtipo do local", categorias["SUBLOCAL"])
with c2:
    periodo = st.selectbox("Período do dia", categorias["PERIODO"])
    dia = st.selectbox("Dia da semana", categorias["DIA_SEMANA"])
    marca = st.selectbox("Marca do celular", categorias["MARCA"])

if st.button("Classificar ocorrência", type="primary", use_container_width=True):
    entrada = {"BAIRRO": bairro, "LOGRADOURO": logradouro, "PERIODO": periodo,
               "SUBLOCAL": sublocal, "MARCA": marca, "DIA_SEMANA": dia}
    p_roubo = float(modelo.predict(vetor(entrada, colunas), verbose=0).ravel()[0])
    p_furto = 1.0 - p_roubo
    classe = "ROUBO" if p_roubo >= 0.5 else "FURTO"

    st.subheader(f"Classe estimada: {classe}")
    m1, m2 = st.columns(2)
    m1.metric("Probabilidade de Roubo", f"{p_roubo*100:.1f}%")
    m2.metric("Probabilidade de Furto", f"{p_furto*100:.1f}%")
    st.progress(min(max(p_roubo, 0.0), 1.0), text=f"Roubo: {p_roubo*100:.1f}%")
    st.caption(
        f"Interpretação: considerando que uma ocorrência com essas características aconteceu, o modelo atribui "
        f"{p_roubo*100:.1f}% de probabilidade à classe *roubo* e {p_furto*100:.1f}% à classe *furto*. "
        "Isto **não** é a chance de um roubo acontecer nesse local/horário."
    )

with st.expander("Sobre o modelo"):
    if meta:
        t = meta.get("teste", {})
        st.write(f"**Arquitetura:** {meta.get('modelo_escolhido')} · **Entradas (One-Hot):** {meta.get('n_entradas')}")
        st.write(f"**Divisão dos dados:** 70% treino / 15% validação / 15% teste")
        st.write(f"**Desempenho no teste (15%):** acurácia {t.get('acuracia', 0)*100:.1f}% · AUC {t.get('auc', 0):.3f}")
    st.write("Atributos usados no treino: bairro, logradouro, período do dia, subtipo do local, marca do celular e dia da semana. "
             "Campos que revelam a qualificação penal foram removidos para evitar vazamento de dados.")
