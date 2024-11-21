from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os


app = FastAPI()

#Configuração do middleware para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Essa classe valida os dados de entrada usando Pydantic
class DadosPrevisao(BaseModel):
    Ano: int
    Municipio: str
    Total_Crimes: float
    serie_1_matriculas: float
    serie_2_matriculas: float
    serie_3_matriculas: float
    serie_4_matriculas: float
    Anos_finais_docentes: float
    Anos_finais_escolas: float
    Anos_iniciais_docentes: float
    Anos_iniciais_escolas: float
    Creche_docentes: float
    Creche_escolas: float
    Creche_matriculas: float
    Ensino_fundamental_docentes: float
    Ensino_fundamental_escolas: float
    Ensino_fundamental_matriculas: float
    Ensino_infantil_docentes: float
    Ensino_infantil_escolas: float
    Ensino_infantil_matriculas: float
    Ensino_medio_docentes: float
    Ensino_medio_escolas: float
    Ensino_medio_matriculas: float
    Estadual_docentes: float
    Estadual_escolas: float
    Estadual_matriculas: float
    Federal_docentes: float
    Federal_escolas: float
    Federal_matriculas: float
    Municipal_docentes: float
    Municipal_escolas: float
    Municipal_matriculas: float
    Nao_seriada_matriculas: float
    Privado_docentes: float
    Privado_escolas: float
    Privado_matriculas: float
    Pre_escolar_docentes: float
    Pre_escolar_escolas: float
    Pre_escolar_matriculas: float
    IDEB: float

#modelo pkl de previsão
MODELO_PATH = '../../models/previsaocrimes.pkl'

#carrega o modelo
try:
    modelo = joblib.load(MODELO_PATH)
    colunas_modelo = getattr(modelo, "feature_names_in_", None)  # Captura as colunas esperadas
    if colunas_modelo is None:
        raise ValueError("O modelo não possui o atributo 'feature_names_in_'.")
except Exception as e:
    modelo = None
    colunas_modelo = []
    print(f"Erro ao carregar o modelo: {e}")

# Rota de teste
@app.get("/")
def home():
    return {"mensagem": "API de Previsão com FastAPI está funcionando!"}

# Rota para previsão
@app.post("/previsao/")
def realizar_previsao(dados: DadosPrevisao):
    if not modelo:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    # Transformar os dados recebidos em um DataFrame
    entrada = pd.DataFrame([dados.dict()])
    
    # Garantir que todas as colunas do modelo estão presentes
    if colunas_modelo:
        for coluna in colunas_modelo:
            if coluna not in entrada.columns:
                entrada[coluna] = 0  # Preencher colunas ausentes com 0
        entrada = entrada[colunas_modelo]  # Reordenar as colunas conforme esperado pelo modelo
    
    # Realizar a previsão
    try:
        previsao = modelo.predict(entrada)
        previsao = max(0, previsao[0])  # Garantir que o valor não seja negativo
        return {"previsao": round(previsao, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a previsão: {e}")
