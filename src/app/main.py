from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import pandas as pd

# Tentativa de carregar o modelo
try:
    modelo_path = '../../models/previsaocrimes.pkl'
    modelo = joblib.load(open(modelo_path, 'rb'))
except Exception as e:
    modelo = None
    print(f"Erro ao carregar o modelo: {e}")

# Caminho do arquivo CSV
csv_path = '../../data/data_set.csv'

app = Flask(__name__)

@app.route('/')
def home():
    return "Minha primeira API."

# Formulário HTML para o usuário preencher
@app.route('/formulario/', methods=['GET', 'POST'])
def formulario():
    if request.method == 'POST':
        try:
            # Obtendo os dados do formulário
            ano = int(request.form['Ano'])
            municipio = request.form['Municipio']

            # Verificar se o arquivo CSV existe
            if not os.path.exists(csv_path):
                return render_template_string(FORM_TEMPLATE, resultado="Erro: Arquivo CSV não encontrado.")
            
            # Carregar os dados do CSV
            dados_csv = pd.read_csv(csv_path)

            # Filtrar os dados do CSV com base nos valores fornecidos
            dados_filtrados = dados_csv[
                (dados_csv['Ano'] == ano) & 
                (dados_csv['Municipio'].str.lower() == municipio.lower())
            ]

            # Verificar se há dados suficientes para previsão
            if dados_filtrados.empty:
                return render_template_string(FORM_TEMPLATE, resultado="Erro: Nenhum dado correspondente encontrado no CSV.")

            # Preparar os dados para previsão
            dados_filtrados = dados_filtrados.drop(columns=['Ano', 'Municipio'])  # Remover colunas desnecessárias

            # Realizando a previsão
            if modelo:
                previsoes = modelo.predict(dados_filtrados)
                resultado = f"Previsão de Crimes e Matrículas: {previsoes[0]}"
                return render_template_string(FORM_TEMPLATE, resultado=resultado)
            else:
                return render_template_string(FORM_TEMPLATE, resultado="Erro: Modelo não carregado.")
        except Exception as e:
            return render_template_string(FORM_TEMPLATE, resultado=f"Erro ao processar os dados: {e}")

    return render_template_string(FORM_TEMPLATE, resultado=None)

# Template HTML para o formulário
FORM_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previsão de Crimes</title>
</head>
<body>
    <h1>Previsão de Crimes e Matrículas</h1>
    <form method="POST">
        <label for="Ano">Ano:</label>
        <input type="number" id="Ano" name="Ano" required><br>
        <label for="Municipio">Municipio:</label>
        <input type="text" id="Municipio" name="Municipio" required><br>
        <button type="submit">Enviar</button>
    </form>
    {% if resultado %}
        <h2>{{ resultado }}</h2>
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
