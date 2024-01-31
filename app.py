#  Aplicativo de previsão de ações com Python e Prophet (Streamlit)

import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go

data_inicio = '2017-01-01'
data_fim = date.today().strftime('%Y-%m-%d')

st.title('Análise de ações')

#Criando a sidebar
st.sidebar.header('Escolha uma opção')

n_dias = st.slider('Quantidade de dias de previsão', 30, 365)

def pegar_dados_acoes():
    path = 'acoes.csv'
    return pd.read_csv(path, delimiter=';')
df = pegar_dados_acoes()

acao = df['snome']
nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação:', acao)

def_acao = df[df['snome'] == nome_acao_escolhida]

acao_escolhida = def_acao.iloc[0]['sigla_acao']

acao_escolhida = acao_escolhida + '.SA'

@st.cache_data
def pegar_valores_online(sigla_acao):
    df = yf.download(sigla_acao, data_inicio, data_fim)
    df.reset_index(inplace=True)
    return df

df_valores = pegar_valores_online(acao_escolhida)

st.subheader('Tabela de valores - '+ nome_acao_escolhida)
st.write(df_valores.tail(11))

#Criar gráfico

st.subheader('Gráfico de Preços')

fig = go.Figure()
fig.add_trace(go.Scatter(x = df_valores['Date'],
                         y = df_valores['Close'],
                         name = 'Preco Fechamento',
                         line_color='yellow'))
fig.add_trace(go.Scatter(x = df_valores['Date'],
                         y = df_valores['Open'],
                         name = 'Preco Abertura',
                         line_color='blue'))
st.plotly_chart(fig)
                       
#previsão

df_treino = df_valores[['Date', 'Close']]

#renomear colunas
df_treino = df_treino.rename(columns = {'Date': 'ds', 'Close': 'y'})
                       
#criando o modelo
modelo = Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe(periods = n_dias, freq = 'B')
previsao = modelo.predict(futuro)

st.subheader('Previsão de Ativos')
st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

#gráfico

grafico1 = plot_plotly(modelo, previsao)
st.plotly_chart(grafico1)

#gráfico2 
grafico2 = plot_components_plotly(modelo, previsao)
st.plotly_chart(grafico2)