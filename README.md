# streamlit-ibge
Criação de dashboard com gráficos interativos sobre os dados extraídos do IBGE.

link para o site: https://app-ibge-projecao-de-alvos.streamlit.app/

## Como executar?

1. Clonar repositório

2. Crie um ambiente virtual:
```
python3 -m venv st
```

3. Ativar o novo ambiente (WSL):
```
source ./st/bin/activate
```

4. Instalar dependências:
```
pip install -r requirements.txt
```

5. Executar o streamlit:
```
streamlit run streamlit_app.py
```
Este comando vai mostrar o seguinte:
```
Local URL: http://localhost:8501
```
Acessar a URL no navegador.