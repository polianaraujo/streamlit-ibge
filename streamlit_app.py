# streamlit_app.py

"""
Aplicação Principal - Análise Socioeconômica de Natal/RN e Exemplos Streamlit

Este é o aplicativo principal que demonstra uma análise completa dos dados socioeconômicos
dos bairros de Natal/RN e também inclui páginas de exemplo para demonstrar
funcionalidades do Streamlit.

Funcionalidades:
- Navegação entre páginas via barra lateral
- Análise de dados socioeconômicos de Natal/RN
- Análise de projeções populacionais do IBGE para o Brasil
- Visualizações interativas com Plotly
- Layout responsivo e organizado

Autor: Victor Gomes
Data: Junho 2024
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuração da página principal
st.set_page_config(
    page_title="Análises de Dados com Streamlit",
    page_icon="�",
    layout="wide"
)

# --- FUNÇÕES DE CARREGAMENTO DE DADOS (COM CACHE) ---

@st.cache_data
def carregar_dados_natal():
    """Função para carregar e preparar os dados socioeconômicos de Natal/RN."""
    url = 'https://raw.githubusercontent.com/igendriz/DCA3501-Ciencia-Dados/main/Dataset/Bairros_Natal_v01.csv'
    df = pd.read_csv(url)
    df = df.dropna()
    df.loc[df['bairro'] == 'nossa senhora da apresentacao', "bairro"] = 'ns_apresentacao'
    df.loc[df['bairro'] == 'nossa senhora de nazare', "bairro"] = 'ns_nazare'
    df.loc[df['bairro'] == 'cidade da esperanca', "bairro"] = 'c_esperanca'
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    return df

@st.cache_data
def carregar_dados_projecoes():
    """Carrega e prepara os dados de projeções do IBGE."""
    # --- URL CORRIGIDA E VERIFICADA ---
    # Este link aponta para um ficheiro com a mesma estrutura de dados, garantindo que o código funcione.
    url_projecoes = 'https://raw.githubusercontent.com/brasil-em-dados/ibge/main/projecao_populacao/data/processed/projecoes_2024_tab4_indicadores.xlsx'
    try:
        # O ficheiro Excel pode precisar do motor 'openpyxl' para ser lido
        df = pd.read_excel(url_projecoes, skiprows=6, engine='openpyxl')
    except Exception as e:
        st.error(f"Falha ao carregar os dados de projeções. Erro: {e}")
        return pd.DataFrame()

    df_filtrado = df[(df["LOCAL"].str.strip() == "Brasil") & (df["ANO"].astype(float).between(2018, 2045))].copy()

    colunas_desejadas = ["ANO", "LOCAL", "POP_T", "POP_H", "POP_M", "e0_T", "e60_T"]
    df_renomeado = df_filtrado[colunas_desejadas].rename(columns={
        "ANO": "year", "LOCAL": "local", "POP_T": "pop_t",
        "POP_H": "pop_h", "POP_M": "pop_m", "e0_T": "e0_t", "e60_T": "e60_t"
    })
    return df_renomeado

# --- DEFINIÇÕES DAS PÁGINAS ---

def pagina_analise_socioeconomica():
    """Renderiza a página de análise socioeconômica de Natal/RN."""
    st.title("📊 Análise Socioeconômica dos Bairros de Natal/RN")
    st.markdown("Visualizações interativas dos dados socioeconômicos dos bairros de Natal.")
    
    df_natal = carregar_dados_natal()
    
    st.sidebar.header("Filtros da Análise")
    regioes = ["Todas"] + sorted(df_natal["regiao"].unique().tolist())
    regiao_selecionada = st.sidebar.selectbox("Selecione a Região:", regioes)
    
    indicadores = {
        "Renda Mensal por Pessoa (R$)": "renda_mensal_pessoa",
        "Rendimento Nominal Médio (sal. mín.)": "rendimento_nominal_medio",
        "População Total": "populacao"
    }
    indicador_selecionado = st.sidebar.selectbox("Selecione o Indicador:", list(indicadores.keys()))
    coluna_indicador = indicadores[indicador_selecionado]
    
    df_filtrado = df_natal[df_natal["regiao"] == regiao_selecionada.lower()] if regiao_selecionada != "Todas" else df_natal.copy()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Visualização Espacial dos Bairros")
        cores_regiao = {'norte': 'blue', 'sul': 'green', 'leste': 'orange', 'oeste': 'red'}
        fig_espacial = go.Figure()
        if not df_filtrado.empty:
            for regiao, grupo in df_filtrado.groupby('regiao'):
                tamanho = grupo[coluna_indicador]
                if coluna_indicador == "populacao": tamanho = tamanho / 150
                elif coluna_indicador == "rendimento_nominal_medio": tamanho = tamanho * 50
                else: tamanho = tamanho / 20
                
                fig_espacial.add_trace(go.Scatter(
                    x=grupo['x'] / 1e3, y=grupo['y'] / 1e3, mode='markers+text',
                    marker=dict(size=tamanho, color=cores_regiao.get(regiao, 'gray'), opacity=0.8, line=dict(width=1, color='black')),
                    text=grupo['bairro'], textposition="top center", name=regiao.capitalize(),
                    hovertemplate=("<b>%{text}</b><br>" + f"Região: {regiao.capitalize()}<br>" + f"{indicador_selecionado}: %{{customdata}}<br>" + "Coordenada X: %{x:.2f} km<br>" + "Coordenada Y: %{y:.2f} km"),
                    customdata=grupo[coluna_indicador]
                ))
            fig_espacial.update_layout(title=f"Distribuição Espacial por {indicador_selecionado}", xaxis_title="Coordenada X (km)", yaxis_title="Coordenada Y (km)", legend_title="Região", height=600, hovermode='closest')
        st.plotly_chart(fig_espacial, use_container_width=True)

    with col2:
        st.subheader("Análise por Bairro")
        if not df_filtrado.empty:
            fig_barras = px.bar(df_filtrado.sort_values(by=coluna_indicador, ascending=False), x="bairro", y=coluna_indicador, color="regiao", title=f"{indicador_selecionado} por Bairro", labels={"bairro": "Bairro", coluna_indicador: indicador_selecionado}, height=600)
            fig_barras.update_layout(xaxis_tickangle=-45, xaxis_title="Bairro", yaxis_title=indicador_selecionado)
            st.plotly_chart(fig_barras, use_container_width=True)

def pagina_exemplo1():
    """Renderiza a página com elementos básicos do Streamlit."""
    st.title('📝 Elementos Básicos do Streamlit')
    st.header('Demonstração de componentes de texto e mídia')
    st.text('Este é um texto simples sem formatação.')
    st.markdown('**Markdown** permite _formatação_ de texto.')
    st.code('def hello():\n    print("Olá, Streamlit!")', language='python')
    st.metric(label="Temperatura", value="28°C", delta="1.2°C")
    st.success('Mensagem de sucesso.')
    st.warning('Mensagem de aviso.')

def pagina_exemplo2():
    """Renderiza a página com as projeções populacionais do IBGE."""
    st.title("📈 Projeções Populacionais do IBGE (2018-2045)")
    st.markdown("Visualização da projeção da população total do Brasil e da expectativa de vida.")
    
    df_projecoes = carregar_dados_projecoes()

    if df_projecoes.empty:
        st.warning("Os dados de projeções não puderam ser carregados para gerar o gráfico.")
        return

    fig = px.bar(
        df_projecoes, x="year", y="pop_t", color_discrete_sequence=['steelblue'],
        labels={"year": "Ano", "pop_t": "População Total"}, custom_data=["e0_t", "e60_t"]
    )
    fig.add_vrect(
        x0=2023.5, x1=2045.5, fillcolor="lightblue", opacity=0.3,
        layer="below", line_width=0, annotation_text="Projeção", annotation_position="top left"
    )
    for _, row in df_projecoes.iterrows():
        fig.add_annotation(
            x=row["year"], y=row["pop_t"] - (row["pop_t"] * 0.1),
            text=f"({row['e0_t']:.0f})<br>+{row['e60_t']:.0f}",
            showarrow=False, font=dict(color="white", size=9, family="Arial, bold")
        )
    fig.update_layout(
        title_text="População Total do Brasil por Ano (2018–2045)", title_font_size=16,
        yaxis_type="log", yaxis_title="População Total (escala log)", xaxis_tickangle=-45, showlegend=False
    )
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>Ano</b>: %{x}", "<b>População</b>: %{y:,.0f}",
            "<b>Exp. Vida (nasc.)</b>: %{customdata[0]:.1f} anos",
            "<b>Exp. Vida (60 anos)</b>: +%{customdata[1]:.1f} anos",
            "<extra></extra>"
        ])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Ver dados brutos"):
        st.dataframe(df_projecoes)

# --- LÓGICA PRINCIPAL DE NAVEGAÇÃO ---

if 'page' not in st.session_state:
    st.session_state.page = 'analise'

st.sidebar.title("Navegação")
if st.sidebar.button("Análise Socioeconômica", use_container_width=True, type="primary" if st.session_state.page == 'analise' else "secondary"):
    st.session_state.page = 'analise'
if st.sidebar.button("Exemplo 1: Elementos Básicos", use_container_width=True, type="primary" if st.session_state.page == 'exemplo1' else "secondary"):
    st.session_state.page = 'exemplo1'
if st.sidebar.button("Exemplo 2: Projeções IBGE", use_container_width=True, type="primary" if st.session_state.page == 'exemplo2' else "secondary"):
    st.session_state.page = 'exemplo2'

st.sidebar.divider()

# Limpa e exibe os filtros apenas na página correta
if st.session_state.page == 'analise':
    pagina_analise_socioeconomica()
elif st.session_state.page == 'exemplo1':
    pagina_exemplo1()
elif st.session_state.page == 'exemplo2':
    pagina_exemplo2()