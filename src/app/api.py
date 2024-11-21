from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json

app = FastAPI()

@app.get("/dados")
def get_dados():
    try:
        # Carregar os dados JSON
        with open('dados.json', 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return {"error": "Arquivo 'dados.json' não encontrado"}
