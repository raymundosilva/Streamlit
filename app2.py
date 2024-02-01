import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go

# Título do aplicativo
st.title('Análise de ações')

# Sidebar
st.sidebar.header('Escolha uma opção')
n_dias = st.sidebar.slider('Quantidade de dias de previsão', 30, 365)

# Função para pegar dados das ações
@st.cache_data
def pegar_dados_acoes():
    path = 'acoes.csv'
    return pd.read_csv(path, delimiter=';')

# Seleção da ação
df = pegar_dados_acoes()
acao = df['snome']
nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação:', acao)
def_acao = df[df['snome'] == nome_acao_escolhida]
acao_escolhida = def_acao.iloc[0]['sigla_acao'] + '.SA'

# Seleção da data
data_inicio = st.sidebar.date_input('Data de início', value=pd.to_datetime('2017-01-01'))
data_fim = st.sidebar.date_input('Data de fim', value=pd.to_datetime(date.today().strftime('%Y-%m-%d')))

# Função para pegar valores online
@st.cache_data
def pegar_valores_online(sigla_acao, data_inicio, data_fim):
    df = yf.download(sigla_acao, data_inicio, data_fim)
    df.reset_index(inplace=True)
    return df

# Mostrando os valores
df_valores = pegar_valores_online(acao_escolhida, data_inicio, data_fim)
st.subheader(f'Valores da ação - {nome_acao_escolhida}')
st.write(df_valores.tail(11).rename(columns={'Date': 'Data', 'Open': 'Abertura', 'High': 'Alta', 'Low': 'Baixa', 'Close': 'Fechamento', 'Adj Close': 'Fechamento Ajustado', 'Volume': 'Volume'}))


# Gráfico de preços
st.subheader('Gráfico de Preços')
fig = go.Figure(data=[
    go.Bar(name='Preço Fechamento', x=df_valores['Date'], y=df_valores['Close']),
    go.Bar(name='Preço Abertura', x=df_valores['Date'], y=df_valores['Open'])
])
fig.update_layout(barmode='group')
st.plotly_chart(fig)

# Previsão
df_treino = df_valores[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

if len(df_treino.dropna()) >= 2:
    modelo = Prophet()
    modelo.fit(df_treino)
    futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
    previsao = modelo.predict(futuro)

    # Mostrando a previsão
    st.subheader('Previsão de Ativos')
    st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias).rename(columns={'ds': 'Data', 'yhat': 'Previsão', 'yhat_lower': 'Previsão Inferior', 'yhat_upper': 'Previsão Superior'}))

    # Gráficos de previsão
    st.subheader('Gráfico de Previsão')
    grafico1 = plot_plotly(modelo, previsao)
    st.plotly_chart(grafico1)

    st.subheader('Componentes da Previsão')
    grafico2 = plot_components_plotly(modelo, previsao)
    st.plotly_chart(grafico2)
else:
    st.error('Não há dados suficientes para fazer uma previsão.')
