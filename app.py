import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Roubo vs Furto - SP", page_icon="📱")

# 1. Carregar modelo, colunas do treino e categorias dos campos
@st.cache_resource
def carregar_arquivos():
    modelo = load_model('modelo_mlp.h5', compile=False)
    with open('colunas_treino.pkl', 'rb') as f:
        colunas = pickle.load(f)
    with open('categorias_dropdown.pkl', 'rb') as f:
        categorias = pickle.load(f)
    return modelo, colunas, categorias

modelo, colunas_treino, categorias = carregar_arquivos()

# 2. Interface
st.title("Classificador de Ocorrências SP: Roubo vs Furto")
st.warning(
    "A MLP não prevê se um crime acontecerá em determinado local ou horário. "
    "Ela analisa as características de uma ocorrência (já registrada ou hipotética) "
    "e indica se o registro é mais compatível com roubo ou furto."
)

st.header("Insira os dados da ocorrência")
st.caption("Os campos listam apenas categorias vistas no treino. \"OUTROS\" agrupa as categorias raras.")
bairro = st.selectbox("Bairro", categorias['BAIRRO_T'])
logradouro = st.selectbox("Logradouro (rua/avenida)", categorias['LOGRADOURO_T'])
sublocal = st.selectbox("Tipo de local", categorias['SUBLOCAL_T'])
periodo = st.selectbox("Período do dia", categorias['PERIODO'])
dia = st.selectbox("Dia da semana", categorias['DIA_SEMANA'])
marca = st.selectbox("Marca do celular", categorias['MARCA_T'])

# 3. Predição
if st.button("Classificar Ocorrência"):
    entrada_df = pd.DataFrame(0.0, index=[0], columns=colunas_treino)

    escolhas = {
        'BAIRRO_T': bairro,
        'LOGRADOURO_T': logradouro,
        'SUBLOCAL_T': sublocal,
        'PERIODO': periodo,
        'DIA_SEMANA': dia,
        'MARCA_T': marca,
    }
    for atributo, valor in escolhas.items():
        coluna = f"{atributo}_{valor}"
        if coluna in entrada_df.columns:
            entrada_df.at[0, coluna] = 1.0
        else:
            st.error(f"Categoria não encontrada no treino: {coluna}")
            st.stop()

    prob_roubo = float(modelo.predict(entrada_df, verbose=0)[0][0])
    prob_furto = 1 - prob_roubo
    classe = "Roubo" if prob_roubo >= 0.5 else "Furto"

    st.subheader("Resultado da Classificação")
    st.success(f"Classe estimada: **{classe}**")
    st.write(f"**Probabilidade de Roubo:** {prob_roubo * 100:.2f}%")
    st.write(f"**Probabilidade de Furto:** {prob_furto * 100:.2f}%")
    st.progress(prob_roubo)

    st.info(
        f"Interpretação correta: considerando que uma ocorrência aconteceu com essas "
        f"características, o modelo atribui {prob_roubo * 100:.2f}% de probabilidade à classe "
        f"'roubo' e {prob_furto * 100:.2f}% à classe 'furto'. Isto não é a chance de um "
        f"crime ocorrer nesse local ou horário."
    )
