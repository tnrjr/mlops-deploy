#Modelo breve api
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Carregar o modelo
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "API de Previsão - Modelo de Machine Learning"

# Endpoint para fazer previsões
@app.route('/predict', methods=['POST'])
def predict():
    # Receber dados JSON do cliente
    data = request.get_json()
    try:
        # Extração dos dados do JSON
        input_data = np.array(data["inputs"]).reshape(1, -1)  # Garantir que o input seja 2D
        prediction = model.predict(input_data)  # Fazer a previsão
        return jsonify({"prediction": prediction[0]})  # Retornar a previsão
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
