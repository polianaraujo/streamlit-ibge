import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from ipywidgets import interact
import ipywidgets as widgets

df_projecoes = pd.read_excel('./tables/projecoes_2024_tab4_indicadores.xlsx', skiprows=6)
df_projecoes

df_projecoes_filtrado = df_projecoes[ (df_projecoes["LOCAL"].str.strip() == "Brasil") & ( df_projecoes["ANO"].astype(float).between(2018, 2045) ) ]
df_projecoes_filtrado

colunas_desejadas = ["ANO", "LOCAL", "POP_T", "POP_H", "POP_M", "e0_T", "e60_T"]
df_projecoes_filtrado = df_projecoes_filtrado[colunas_desejadas].rename(columns={
    "ANO": "year",
    "LOCAL": "local",
    "POP_T": "pop_t",
    "POP_H": "pop_h",
    "POP_M": "pop_m",
    "e0_T": "e0_t",
    "e60_T": "e60_t"
})

df_projecoes_filtrado

# Salvar CSV
# projecoes2_df_filtrado.to_csv(os.path.join(csv_path,"projecoes.csv"), index=False)

# Criar o gráfico de barras
plt.figure(figsize=(22, 8))

anos = df_projecoes_filtrado["year"].tolist()
inicio = anos.index(2024)
fim = anos.index(2045)
plt.axvspan(inicio - 0.5, fim + 0.5, color="lightblue", alpha=0.4, label="Projeção")

ax = sns.barplot(data=df_projecoes_filtrado, x="year", y="pop_t", color="steelblue")

# Adicionar texto em cada barra
for index, row in df_projecoes_filtrado.iterrows():
    ano = row["year"]
    pop_total = row["pop_t"]
    e0_t = row["e0_t"]
    e60_t = row["e60_t"]

    plt.text(
        x = index - 18,
        y = pop_total - 6e5,
        s=f"({e0_t:,.0f}) \n (+{e60_t:,.0f})",
        ha = "center",
        fontsize = 9.5,
        color = "yellow",
        fontweight = "bold",
        label="(e0_t) | (+e60_t)"
    )

# Personalização
plt.title("População Total do Brasil por Ano (2018–2045)", fontsize=14)
plt.xlabel("Ano", fontsize=12)
plt.ylabel("População Total", fontsize=12)
plt.yscale("log")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
