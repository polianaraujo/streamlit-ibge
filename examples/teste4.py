# streamlit_app.py

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
    anos = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
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



@st.cache_data
def carregar_dados_de_renda():
    """
    Carrega e unifica os dados de renda por instrução (hora) e por idade (mês)
    a partir de URLs do IBGE, retornando um único DataFrame.
    """
    url_salario_etario = 'https://raw.githubusercontent.com/polianaraujo/streamlit-ibge/main/tables/tabela_1_15_OcupCaract_Geo_Rend.xls'
    url_salario_instrucao = 'https://raw.githubusercontent.com/polianaraujo/streamlit-ibge/main/tables/tabela_1_17_InstrCaract_Rend.xls'
    anos = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]

    # --- 1. Carregar dados de Renda por Instrução (inst_sal) ---
    try:
        colunas_desejadas_BR_inst = {"Grandes Regiões, sexo e cor ou raça": "BR"}
        colunas_desejadas_INST = {
            "Sem instrução ou fundamental incompleto": "incomplete",
            "Ensino fundamental completo ou médio incompleto": "elementary",
            "Ensino médio completo ou superior incompleto": "high",
            "Ensino superior completo": "college"
        }
        
        lista_dfs_inst = []
        for ano in anos:
            df_br = pd.read_excel(
                url_salario_instrucao, sheet_name=ano, skiprows=3,
                usecols=list(colunas_desejadas_BR_inst.keys()), engine='xlrd'
            ).rename(columns=colunas_desejadas_BR_inst).iloc[[4]].reset_index(drop=True)

            df_vals = pd.read_excel(
                url_salario_instrucao, sheet_name=ano, skiprows=5, engine='xlrd'
            ).drop(columns=["Unnamed: 0", "Unnamed: 1"]).drop([0,1]).iloc[[0]].reset_index(drop=True).rename(columns=colunas_desejadas_INST)
            
            df_br['year'] = ano
            df_ano_completo = df_br.join(df_vals)
            lista_dfs_inst.append(df_ano_completo)
            
        inst_sal = pd.concat(lista_dfs_inst, ignore_index=True)

    except Exception as e:
        st.error(f"Erro ao carregar dados de renda por instrução. Erro: {e}")
        return pd.DataFrame()

    # --- 2. Carregar dados de Renda por Faixa Etária (idade_sal) ---
    try:
        colunas_desejadas_IDADE_BR = {"Grandes Regiões, Unidades da Federação e Municípios das Capitais": "BR"}
        colunas_desejadas_IDADE = ["14 a 29 anos", "30 a 49 anos", "50 a 59 anos", "60 anos ou mais"]

        lista_dfs_idade = []
        for ano in anos:
            df_br_idade = pd.read_excel(
                url_salario_etario, sheet_name=ano, skiprows=2,
                usecols=list(colunas_desejadas_IDADE_BR.keys()), engine='xlrd'
            ).rename(columns=colunas_desejadas_IDADE_BR).iloc[[3]].reset_index(drop=True)
            
            df_vals_idade = pd.read_excel(
                url_salario_etario, sheet_name=ano, skiprows=4,
                usecols=colunas_desejadas_IDADE, engine='xlrd'
            ).iloc[[1]].reset_index(drop=True)
            
            df_br_idade['year'] = ano
            df_ano_idade_completo = df_br_idade.join(df_vals_idade)
            lista_dfs_idade.append(df_ano_idade_completo)
        
        idade_sal = pd.concat(lista_dfs_idade, ignore_index=True)
        
        # ✨ CORREÇÃO: Renomeia as colunas de idade para nomes padronizados,
        # que serão usados pela função que cria os gráficos.
        idade_sal = idade_sal.rename(columns={
            "14 a 29 anos": "rend_mes_14_29",
            "30 a 49 anos": "rend_mes_30_49",
            "50 a 59 anos": "rend_mes_50_59",
            "60 anos ou mais": "rend_mes_60_mais"
        })

    except Exception as e:
        st.error(f"Erro ao carregar dados de renda por idade. Erro: {e}")
        return pd.DataFrame()

    # --- 3. Unificar os DataFrames ---
    df_final_unificado = pd.merge(inst_sal, idade_sal, on=["BR", "year"], how="inner")
    
    return df_final_unificado




# --- DEFINIÇÕES DAS PÁGINAS ---

# def pagina_exemplo1():
#     """Renderiza a página com elementos básicos do Streamlit."""
#     st.title('📝 Elementos Básicos do Streamlit')
#     st.header('Demonstração de componentes de texto e mídia')
#     st.text('Este é um texto simples sem formatação.')
#     st.markdown('**Markdown** permite _formatação_ de texto.')
#     st.code('def hello():\n    print("Olá, Streamlit!")', language='python')
#     st.metric(label="Temperatura", value="28°C", delta="1.2°C")
#     st.success('Mensagem de sucesso.')
#     st.warning('Mensagem de aviso.')

#adicionado
def pagina_capa_dashboard():
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #4a90e2; font-size: 48px;">Dashboard IBGE - Brasil 2018-2045</h1>
            <p style="font-size: 20px; color: #555555;">
                Um painel interativo com dados de <strong>população</strong>, <strong>força de trabalho</strong> e <strong>rendimento</strong>, 
                baseado em análises do IBGE.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        ### 🔍 O que você encontrará aqui?
        - 📈 Projeções populacionais do Brasil de 2018 a 2045
        - 👨‍👩‍👧‍👦 Evolução da força de trabalho por faixa etária e grau de instrução
        - 💵 Análises de rendimento médio por hora e por mês
        - 🎬 Gráficos animados para visualizar tendências ao longo dos anos

        ---
    """)

    st.markdown("""
        <div style="background-color: #d9edf7; padding: 10px; border-radius: 5px;">
            <span style="color: #31708f; font-size: 16px;">💡 <strong>Use o menu lateral para navegar pelas análises.</strong></span>
        </div>
    """, unsafe_allow_html=True)


    st.markdown("""
        <div style="text-align: center; margin-top: 40px;">
            <em>Desenvolvido por Poliana e Rosélia • Ciência de Dados 2025</em>
        </div>
    """, unsafe_allow_html=True)

def pagina_exemplo1():
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
        
def pagina_exemplo2():
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

def pagina_exemplo3():
    """
    Renderiza uma página com um gráfico de área e um de barras animado,
    ambos controlados dinamicamente por um seletor de análise.
    """
    st.title("📈 Evolução da Força de Trabalho (2012-2023)")
    st.markdown(
        "Veja como a força de trabalho no Brasil evoluiu. "
        "Use o seletor para alternar entre a análise por **faixa etária** ou por **grau de instrução**."
    )

    # --- 1. Seletor para o usuário escolher a análise ---
    tipo_analise = st.radio(
        "Escolha o tipo de análise:",
        ("Faixa Etária", "Grau de Instrução"),
        horizontal=True,
    )

    # --- 2. Definição dinâmica dos parâmetros ---
    # As variáveis aqui definidas serão usadas para AMBOS os gráficos.
    if tipo_analise == "Faixa Etária":
        colunas = {"Características selecionadas": "features", "População na força de trabalho\n(1 000 pessoas)": "work_pop"}
        faixas = {"homens": (15, 22), "mulheres": (24, 31)}
        feature_col = "features"  # <-- Esta variável torna o código dinâmico!
        legenda_titulo = "Faixa Etária"
    else: # Grau de Instrução
        colunas = {"Características selecionadas": "degree", "População na força de trabalho\n(1 000 pessoas)": "work_pop"}
        faixas = {"homens": (58, 62), "mulheres": (64, 68)}
        feature_col = "degree"  # <-- Esta variável torna o código dinâmico!
        legenda_titulo = "Grau de Instrução"

    # --- 3. Carregamento dos dados ---
    # O DataFrame 'df' conterá os dados corretos (etários ou de instrução)
    df = carregar_e_processar_dados(colunas, faixas, feature_col)

    if df.empty:
        st.warning("Não foi possível carregar os dados para a análise selecionada.")
        return

    # --- GRÁFICO 1: ÁREA EMPILHADA (Dinâmico) ---
    st.subheader(f"Composição da Força de Trabalho por {legenda_titulo}")
    
    df_agrupado = df.groupby(['year', feature_col])['work_pop'].sum().reset_index()

    fig_area = px.area(
        df_agrupado,
        x='year',
        y='work_pop',
        color=feature_col, # Usa a variável dinâmica
        title=f'Evolução da Força de Trabalho por {legenda_titulo}',
        labels={'year': 'Ano', 'work_pop': 'População', feature_col: legenda_titulo},
        markers=True
    )
    fig_area.update_layout(hovermode="x unified", legend_title=legenda_titulo, yaxis_title="População")
    fig_area.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig_area, use_container_width=True)

    
    st.divider()

    
    # --- GRÁFICO 2: BARRAS ANIMADAS (Dinâmico) ---
    st.subheader(f"Evolução Detalhada por Sexo e {legenda_titulo}")

    df_animado = df.copy()
    df_animado['sex'] = df_animado['sex'].map({'H': 'Homens', 'M': 'Mulheres'})
    
    # A coluna "Grupo" é criada usando a variável dinâmica 'feature_col'
    df_animado["Grupo"] = df_animado["sex"] + " - " + df_animado[feature_col]

    fig_animado = px.bar(
        df_animado,
        x="work_pop",
        y="Grupo",
        color="Grupo",
        orientation="h",
        animation_frame="year",
        animation_group="Grupo",
        range_x=[0, df_animado["work_pop"].max() * 1.1],
        labels={"work_pop": "População na Força de Trabalho", "Grupo": "Grupos"},
        title=f"Evolução por Sexo e {legenda_titulo}" # Título dinâmico
    )
    # Ordena as barras a cada ano para melhor visualização na animação
    fig_animado.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_animado.update_xaxes(showgrid=True)
    st.plotly_chart(fig_animado, use_container_width=True)


    # Expander para ver os dados brutos
    with st.expander("Ver dados brutos da análise"):
        st.dataframe(df)


def pagina_exemplo4():
    """
    Renderiza uma página com as análises de rendimento, combinando gráficos de linha
    com gráficos de caixa para uma visão completa.
    """
    st.title("Análise de Rendimento por Idade e Instrução (2012-2023)")
    st.markdown(
        "Comparação da evolução do rendimento médio no Brasil (gráficos de linha) e da "
        "distribuição desses rendimentos ao longo dos anos (gráficos de caixa)."
    )

    # Carrega o DataFrame unificado
    df_renda_unificado = carregar_dados_de_renda()

    if df_renda_unificado.empty:
        st.warning("Não foi possível carregar os dados de renda para a análise.")
        return

    st.subheader("Tendência do Rendimento Médio ao Longo do Tempo")
    col1, col2 = st.columns(2)

    # --- Gráfico 1: Rendimento por Faixa Etária (na coluna 1) ---
    with col1:
        # Colunas de rendimento mensal por idade
        cols_idade = ["rend_mes_14_29", "rend_mes_30_49", "rend_mes_50_59", "rend_mes_60_mais"]
        
        # Transforma o DF para o formato longo
        df_long_idade = df_renda_unificado.melt(
            id_vars=['year'], value_vars=cols_idade,
            var_name='faixa_etaria', value_name='rendimento_mes'
        ).dropna(subset=['rendimento_mes'])
        
        mapa_nomes_idade = {
            "rend_mes_14_29": "14 a 29 anos", "rend_mes_30_49": "30 a 49 anos",
            "rend_mes_50_59": "50 a 59 anos", "rend_mes_60_mais": "60 anos ou mais"
        }
        df_long_idade['faixa_etaria'] = df_long_idade['faixa_etaria'].map(mapa_nomes_idade)

        fig_etaria = px.line(
            df_long_idade, x='year', y='rendimento_mes', color='faixa_etaria', markers=True,
            labels={"year": "Ano", "rendimento_mes": "Rendimento Médio Mensal (R$)", "faixa_etaria": "Faixa Etária"}
        )
        fig_etaria.update_layout(xaxis_tickangle=-45, legend_title="Faixa Etária", hovermode="x unified")
        fig_etaria.update_traces(hovertemplate='<b>%{data.name}</b><br>Rendimento: R$ %{y:,.2f}<extra></extra>')
        st.plotly_chart(fig_etaria, use_container_width=True)

    # --- Gráfico 2: Rendimento por Grau de Instrução (na coluna 2) ---
    with col2:
        # Colunas de rendimento por hora por instrução
        cols_instrucao = ["incomplete", "elementary", "high", "college"]

        df_long_instrucao = df_renda_unificado.melt(
            id_vars=['year'], value_vars=cols_instrucao,
            var_name='grau_instrucao', value_name='rendimento_hora'
        ).dropna(subset=['rendimento_hora'])
        
        mapa_nomes_instrucao = {
            "incomplete": "Sem instrução ou Fund. Incompleto",
            "elementary": "Fund. Completo ou Médio Incompleto",
            "high": "Médio Completo ou Sup. Incompleto",
            "college": "Superior Completo"
        }
        df_long_instrucao['grau_instrucao'] = df_long_instrucao['grau_instrucao'].map(mapa_nomes_instrucao)

        fig_instrucao = px.line(
            df_long_instrucao, x='year', y='rendimento_hora', color='grau_instrucao', markers=True,
            labels={"year": "Ano", "rendimento_hora": "Rendimento Médio por Hora (R$)", "grau_instrucao": "Grau de Instrução"}
        )
        fig_instrucao.update_layout(xaxis_tickangle=-45, legend_title="Grau de Instrução", hovermode="x unified")
        fig_instrucao.update_traces(hovertemplate='<b>%{data.name}</b><br>Rendimento: R$ %{y:,.2f}<extra></extra>')
        st.plotly_chart(fig_instrucao, use_container_width=True)

    
    # --- NOVA SEÇÃO: GRÁFICOS DE CAIXA ---
    
    st.divider()
    st.subheader("Análise da Distribuição dos Rendimentos (2012-2023)")
    st.markdown(
        "A análise abaixo mostra a variação dos rendimentos médios anuais para cada grupo. "
        "Isso ajuda a entender a **volatilidade** e a **faixa de valores** de cada categoria ao longo do tempo."
    )
    
    col3, col4 = st.columns(2)

    # --- Gráfico 3: Box Plot de Rendimento por Faixa Etária (na coluna 3) ---
    with col3:
        fig_box_idade = px.box(
            df_long_idade,  # Reaproveitando o DataFrame do gráfico de linha
            x='faixa_etaria',
            y='rendimento_mes',
            color='faixa_etaria',
            points="all",
            title="Distribuição do Rend. Mensal por Idade"
        )
        fig_box_idade.update_layout(xaxis_title="Faixa Etária", yaxis_title="Rendimento Mensal (R$)", showlegend=False)
        st.plotly_chart(fig_box_idade, use_container_width=True)

    # --- Gráfico 4: Box Plot de Rendimento por Grau de Instrução (na coluna 4) ---
    with col4:
        fig_box_instrucao = px.box(
            df_long_instrucao, # Reaproveitando o DataFrame do gráfico de linha
            x='grau_instrucao',
            y='rendimento_hora',
            color='grau_instrucao',
            points="all",
            title="Distribuição do Rend. por Hora por Instrução"
        )
        fig_box_instrucao.update_layout(xaxis_title="Grau de Instrução", yaxis_title="Rendimento por Hora (R$)", showlegend=False)
        st.plotly_chart(fig_box_instrucao, use_container_width=True)

    # --- Expander de dados brutos ao final ---
    with st.expander("Ver dados brutos unificados"):
        st.dataframe(df_renda_unificado)



# --- LÓGICA PRINCIPAL DE NAVEGAÇÃO ---

# Define a página inicial se não estiver definida
if 'page' not in st.session_state:
    #modificado também
    st.session_state.page = 'capa_dashboard'

# Barra lateral de navegação
# st.sidebar.title("Navegação")
# if st.sidebar.button("Exemplo 1: Elementos Básicos", use_container_width=True, type="primary" if st.session_state.page == 'exemplo1' else "secondary"):
    # st.session_state.page = 'exemplo1'

#adicionado
if st.sidebar.button("Home", use_container_width=True):
    st.session_state.page = 'capa_dashboard'
if st.sidebar.button("Projeções IBGE", use_container_width=True, type="primary" if st.session_state.page == 'exemplo1' else "secondary"):
    st.session_state.page = 'exemplo1'
if st.sidebar.button("População por Sexo", use_container_width=True, type="primary" if st.session_state.page == 'exemplo2' else "secondary"):
    st.session_state.page = 'exemplo2'
if st.sidebar.button("Análise da Força de Trabalho", use_container_width=True, type="primary" if st.session_state.page == 'exemplo3' else "secondary"):
    st.session_state.page = 'exemplo3'
if st.sidebar.button("Análise de Renda", use_container_width=True, type="primary" if st.session_state.page == 'exemplo4' else "secondary"):
    st.session_state.page = 'exemplo4'

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
elif st.session_state.page == 'capa_dashboard':
    pagina_capa_dashboard()