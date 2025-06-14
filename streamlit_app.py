# streamlit_app.py

"""
Aplicação Principal - Análise Socioeconômica e Exemplos Streamlit

Este é o aplicativo principal que demonstra uma análise completa dos dados socioeconômicos
e também inclui páginas de exemplo para demonstrar funcionalidades do Streamlit.

Funcionalidades:
- Navegação entre páginas via barra lateral
- Análise de projeções populacionais do IBGE para o Brasil
- Análise da força de trabalho por faixa etária e grau de instrução
- Visualizações interativas com Plotly e Matplotlib
- Layout responsivo e organizado

Autor: Victor Gomes
Data: Junho 2024 (com ajustes de Gemini)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página principal
st.set_page_config(
    page_title="Análises de Dados com Streamlit",
    page_icon="📊",
    layout="wide"
)

# --- FUNÇÕES DE CARREGAMENTO DE DADOS (COM CACHE) ---

@st.cache_data
def carregar_dados_projecoes():
    """Carrega e prepara os dados de projeções do IBGE."""
    url_projecoes = 'https://raw.githubusercontent.com/polianaraujo/streamlit-ibge/main/tables/projecoes_2024_tab4_indicadores.xlsx'
    try:
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

@st.cache_data
def carregar_e_processar_dados(colunas_desejadas, faixas, feature_col_name):
    """
    Função genérica para carregar e processar dados do arquivo Excel do IBGE.
    Carrega dados da força de trabalho por faixa etária ou instrução.
    """
    anos = ["2018", "2019", "2020", "2021", "2022", "2023"]
    url = "https://raw.githubusercontent.com/polianaraujo/streamlit-ibge/main/tables/tabela_1_1_Indic_BR.xls"
    
    # Função interna para carregar os dados de uma aba específica (ano)
    def carregar_dados_por_ano(ano):
        df = pd.read_excel(
            url,
            sheet_name=ano,
            skiprows=2,
            usecols=list(colunas_desejadas.keys())
        ).rename(columns=colunas_desejadas)
        df['year'] = ano
        return df

    # Função interna para extrair os subconjuntos (homens/mulheres)
    def processar_subconjuntos(df):
        return {
            faixa: df.iloc[inicio:fim].assign(sex=sexo)
            for faixa, (inicio, fim), sexo in zip(faixas.keys(), faixas.values(), ['H', 'M'])
        }

    # Concatena os dados de todos os anos
    df_final = pd.concat(
        [
            pd.concat(processar_subconjuntos(carregar_dados_por_ano(ano)).values(), axis=0, ignore_index=True)
            for ano in anos
        ],
        axis=0, ignore_index=True
    )

    # Ajustes finais no DataFrame
    df_final = df_final[["year", "sex", feature_col_name, "work_pop"]]
    df_final["work_pop"] *= 1000
    
    # Remove linhas indesejadas (aplica-se apenas à faixa etária)
    if feature_col_name == "features":
        df_final = df_final[~df_final["features"].isin(["14 a 17 anos", "18 a 24 anos", "25 a 29 anos"])]

    return df_final.reset_index(drop=True)


# --- DEFINIÇÕES DAS PÁGINAS ---

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
        
def pagina_exemplo3():
    """Renderiza a página com gráfico interativo da população por sexo."""
    st.title("👨‍👩‍👧‍👦 População por Sexo no Brasil (2018–2045)")
    st.markdown("Visualização da evolução da população **masculina** e **feminina** no Brasil com base nos dados do IBGE.")

    df = carregar_dados_projecoes()
    if df.empty:
        st.warning("Não foi possível carregar os dados para o gráfico.")
        return

    df_long = df.melt(
        id_vars=["year"], 
        value_vars=["pop_h", "pop_m"], 
        var_name="Sexo", 
        value_name="População"
    )
    df_long["Sexo"] = df_long["Sexo"].map({"pop_h": "Masculina", "pop_m": "Feminina"})

    fig = px.line(
        df_long, x="year", y="População", color="Sexo", markers=True,
        labels={"year": "Ano", "População": "População Estimada", "Sexo": "Sexo"},
        title="Evolução da População de Homens e Mulheres no Brasil (2018–2045)",
        color_discrete_map={"Masculina": "blue", "Feminina": "red"}
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="População (milhões)",
        legend_title="Sexo",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver dados brutos"):
        st.dataframe(df[["year", "pop_h", "pop_m"]])

def pagina_exemplo4():
    """Renderiza a página com a análise interativa da população por faixa etária."""
    st.title("📊 Análise da População por Faixa Etária")
    st.markdown("População na força de trabalho por faixa etária, de 2018 a 2023. Passe o mouse sobre as linhas para ver os valores.")

    # Configurações específicas para faixa etária
    colunas_desejadas_etario = {
        "Características selecionadas": "features",
        "População na força de trabalho\n(1 000 pessoas)": "work_pop"
    }
    faixas_etario = {
        "homens": (15, 22),
        "mulheres": (24, 31)
    }
    
    # Carrega os dados usando a função genérica
    etario_filtrado = carregar_e_processar_dados(colunas_desejadas_etario, faixas_etario, "features")

    if etario_filtrado.empty:
        st.warning("Não foi possível carregar os dados para a análise por faixa etária.")
        return

    # Agrupa os dados para o gráfico
    etario_agrupado = etario_filtrado.groupby(["year", "features"])["work_pop"].sum().reset_index()

    # Cria o gráfico de linhas interativo com Plotly
    fig = px.line(
        etario_agrupado,
        x="year",
        y="work_pop",
        color="features",
        markers=True,
        labels={
            "year": "Ano",
            "work_pop": "População na Força de Trabalho",
            "features": "Faixa Etária"
        },
        title="Evolução da Força de Trabalho por Faixa Etária (2018-2023)"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="População (em milhões)",
        legend_title="Faixa Etária",
        hovermode="x unified"
    )
    
    # Formata o eixo Y para exibir em milhões
    fig.update_yaxes(tickformat=".2fM")

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver dados brutos"):
        st.dataframe(etario_filtrado)


def pagina_exemplo5():
    """Renderiza a página com a análise interativa da população por grau de instrução."""
    st.title("🎓 Análise da População por Grau de Instrução")
    st.markdown("População na força de trabalho por grau de instrução, de 2018 a 2023. Passe o mouse sobre as linhas para ver os valores.")

    # Configurações específicas para grau de instrução
    colunas_desejadas_instrucao = {
        "Características selecionadas": "degree",
        "População na força de trabalho\n(1 000 pessoas)": "work_pop"
    }
    faixas_instrucao = {
        "homens": (58, 62),
        "mulheres": (64, 68)
    }
    
    # Carrega os dados usando a função genérica
    socio_filtrado = carregar_e_processar_dados(colunas_desejadas_instrucao, faixas_instrucao, "degree")

    if socio_filtrado.empty:
        st.warning("Não foi possível carregar os dados para a análise por grau de instrução.")
        return

    # Agrupa os dados para o gráfico
    socio_agrupado = socio_filtrado.groupby(['year', 'degree'])['work_pop'].sum().reset_index()

    # Cria o gráfico de linhas interativo com Plotly
    fig = px.line(
        socio_agrupado,
        x="year",
        y="work_pop",
        color="degree",
        markers=True,
        labels={
            "year": "Ano",
            "work_pop": "População na Força de Trabalho",
            "degree": "Grau de Instrução"
        },
        title="Evolução da Força de Trabalho por Grau de Instrução (2018-2023)"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="População (em milhões)",
        legend_title="Grau de Instrução",
        hovermode="x unified"
    )

    # Formata o eixo Y para exibir em milhões
    fig.update_yaxes(tickformat=".2fM")

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver dados brutos"):
        st.dataframe(socio_filtrado)


# --- LÓGICA PRINCIPAL DE NAVEGAÇÃO ---

# Define a página inicial se não estiver definida
if 'page' not in st.session_state:
    st.session_state.page = 'exemplo1'

# Barra lateral de navegação
st.sidebar.title("Navegação")
if st.sidebar.button("Exemplo 1: Elementos Básicos", use_container_width=True, type="primary" if st.session_state.page == 'exemplo1' else "secondary"):
    st.session_state.page = 'exemplo1'
if st.sidebar.button("Exemplo 2: Projeções IBGE", use_container_width=True, type="primary" if st.session_state.page == 'exemplo2' else "secondary"):
    st.session_state.page = 'exemplo2'
if st.sidebar.button("Exemplo 3: População por Sexo", use_container_width=True, type="primary" if st.session_state.page == 'exemplo3' else "secondary"):
    st.session_state.page = 'exemplo3'
if st.sidebar.button("Exemplo 4: Faixa Etária", use_container_width=True, type="primary" if st.session_state.page == 'exemplo4' else "secondary"):
    st.session_state.page = 'exemplo4'
if st.sidebar.button("Exemplo 5: Grau de Instrução", use_container_width=True, type="primary" if st.session_state.page == 'exemplo5' else "secondary"):
    st.session_state.page = 'exemplo5'


st.sidebar.divider()

# Roteamento de página
if st.session_state.page == 'exemplo1':
    pagina_exemplo1()
elif st.session_state.page == 'exemplo2':
    pagina_exemplo2()
elif st.session_state.page == 'exemplo3':
    pagina_exemplo3()
elif st.session_state.page == 'exemplo4':
    pagina_exemplo4()
elif st.session_state.page == 'exemplo5':
    pagina_exemplo5()