# Classificador de Ocorrências em SP: Roubo × Furto (MLP)

Global Solution – Aprendizado Profundo com Redes Neurais Convolucionais.

Uma rede neural **MLP** analisa as características de uma ocorrência (bairro, logradouro, tipo de local, período do dia, dia da semana e marca do celular) e indica se o registro é mais compatível com **roubo** ou **furto**.

> ⚠️ O modelo **não** prevê se um crime vai acontecer. A saída é a probabilidade de cada classe, **dado que a ocorrência já aconteceu** com aquelas características. Não é a chance de um crime ocorrer em determinado local ou horário.

## 🔗 Aplicação web

**https://sp-crime-classification-mlp-gs-redes-neurais.streamlit.app/**

## 👥 Integrantes

| Nome | RM | Turma | 
| ----- | ----- | ----- | 
| Raul Bridi Albano | 567772 | CCPB | 
| Ernandes da Silva Jesus | 566611 | CCPB | 
| Leandro Filippini Aguiar Alves | 568157 | CCPB | 

## 📊 Fonte dos dados

SSP-SP – Portal de Estatísticas → Bases de Dados → *Celulares Subtraídos 2025*:
https://www.ssp.sp.gov.br/estatistica/consultas

**Como obter:** baixar a base no portal e salvar como `CelularesSubtraidos_2025.csv` (separador `;`, codificação `latin1`). O arquivo não está neste repositório por causa do tamanho.

**Recorte e preparação**

* Fatos ocorridos no município de São Paulo (`NOME_MUNICIPIO_CIRC`, local do fato e não do registro do BO): 216.287 registros.

* Um registro por BO (`NOME_DELEGACIA` + `ANO_BO` + `NUM_BO`): 185.088.

* Apenas Roubo (art. 157) e Furto (art. 155): 160.024.

* Base balanceada com todos os roubos e o mesmo número de furtos sorteados: **117.176 registros** (58.588 por classe).

* Rótulo: 0 = Furto, 1 = Roubo. A coluna `RUBRICA` não é usada como entrada.

**Atributos de entrada:** bairro, logradouro, período do dia, subtipo do local, marca do celular e dia da semana (One-Hot, 596 colunas). Categorias raras são agrupadas em `OUTROS`.

**Cuidados contra vazamento de dados**

* `DESCR_CONDUTA`, `DESDOBRAMENTO` e `CIRCUNSTANCIA` foram descartados, pois descrevem a qualificação penal.

* Campos do procedimento de registro (`DESCRICAO_APRESENTACAO`, `AUTORIA_BO`, flags) também foram descartados.

* `DESCR_PERIODO` só vem preenchido quando `HORA_OCORRENCIA` está vazia. Tratar isso como categoria "não informado" separava roubo de furto por um artefato do preenchimento. O período foi então derivado da hora quando ela existe.

## 🧠 Modelo

Quatro arquiteturas foram comparadas (ReLU nas ocultas, saída com 1 neurônio sigmoid, `binary_crossentropy`, Adam com lr 0,001, batch 128, `EarlyStopping` com paciência 4):

| Modelo | Camadas ocultas | Acurácia | Precision | Recall | F1 | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| Arq. 1 (Base) | 64 → 32 | 0,7132 | 0,7335 | 0,6698 | 0,7002 | 
| **Arq. 2 (Dropout)** ✅ | 64 → 32 + Dropout 0,3 | 0,7169 | 0,7319 | 0,6845 | **0,7074** | 
| Arq. 3 (Profunda) | 128 → 64 → 32 + Dropout 0,3 | 0,7139 | 0,7408 | 0,6581 | 0,6970 | 
| Arq. 4 (Larga) | 256 → 128 + Dropout 0,5 | 0,7145 | 0,7354 | 0,6701 | 0,7012 | 

*Métricas no conjunto de validação (17.576 registros). Escolha pelo maior F1 na validação.*

**Avaliação final (teste, 17.577 registros):** acurácia **0,7168** e AUC **0,792**.

| Classe | Precision | Recall | F1 | 
| ----- | ----- | ----- | ----- | 
| Furto | 0,70 | 0,75 | 0,73 | 
| Roubo | 0,73 | 0,68 | 0,71 | 

Divisão: treino 82.023 | validação 17.576 | teste 17.577. As quatro arquiteturas ficaram muito próximas, e o `EarlyStopping` limitou o overfitting (gap treino–validação de 0,006 no modelo escolhido). O desempenho é limitado pela informação dos atributos categóricos, não pela capacidade da rede.

## 📁 Estrutura

| Arquivo | Descrição | 
| ----- | ----- | 
| `GS_MPL_SP_FxR.ipynb` | Preparação dos dados, treino das 4 arquiteturas, métricas, curvas e geração do modelo. | 
| `app.py` | Aplicação web (Streamlit). | 
| `artefatos/modelo_mlp.keras` | Modelo treinado definitivo (Arq. 2). | 
| `artefatos/colunas_treino.json` | Lista exata das 596 colunas de entrada (One-Hot). | 
| `artefatos/categorias.json` | Categorias válidas de cada atributo (usadas nos campos do app). | 
| `requirements.txt` | Dependências (TensorFlow, Streamlit, etc). | 

## ▶️ Como executar a aplicação

```
pip install -r requirements.txt
streamlit run app.py

```

## 🔁 Como reproduzir o treino

1. Baixar a base no portal da SSP-SP e salvar como `CelularesSubtraidos_2025.csv`.

2. Abrir o notebook (Google Colab ou localmente) e ajustar o caminho do arquivo na primeira célula de código.

3. Executar todas as células. Ao final, a pasta `artefatos/` será preenchida com `modelo_mlp.keras`, `colunas_treino.json` e `categorias.json`.

## ⚠️ Limitações

* Categorias com variação de acento/caixa (ex.: REPUBLICA/REPÚBLICA) não foram unificadas, o que divide parte do sinal entre colunas.

* Muitas ocorrências com as mesmas características têm rótulos diferentes (ruído irredutível), o que limita a acurácia.

* A amostra é balanceada (50/50) e não reflete a proporção real entre roubos e furtos na base original (\~36% Roubo / \~63% Furto).

* O modelo classifica o tipo de uma ocorrência; não estima risco de crime em tempo real.

## 🤖 Uso de IA generativa

Usamos IA Claude como apoio na depuração, na revisão do código e dos dados (o que ajudou a identificar vazamentos, como o do período do dia, e a otimizar a exportação de artefatos em JSON). Os resultados foram rigorosamente executados e conferidos pela equipe.
