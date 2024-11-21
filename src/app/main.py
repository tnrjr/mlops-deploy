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


@app.post("/dados/")
def previsao_preco_imovel(DadosPrevisao: DadosPrevisao):

    preco = predict_price(imovel)
    
    return {"precoPrevisto": round(preco, 2)}


def predict_price(dados_previsao: DadosPrevisao):
    # Carregar o modelo
    model = joblib.load('../../models/previsaocrimes.pkl')

    # Criar o DataFrame com base nos atributos da classe DadosPrevisao
    dados = pd.DataFrame([{
        'Ano': dados_previsao.Ano,
        'Municipio': dados_previsao.Municipio,
        'Total_Crimes': dados_previsao.Total_Crimes,
        'serie_1_matriculas': dados_previsao.serie_1_matriculas,
        'serie_2_matriculas': dados_previsao.serie_2_matriculas,
        'serie_3_matriculas': dados_previsao.serie_3_matriculas,
        'serie_4_matriculas': dados_previsao.serie_4_matriculas,
        'Anos_finais_docentes': dados_previsao.Anos_finais_docentes,
        'Anos_finais_escolas': dados_previsao.Anos_finais_escolas,
        'Anos_iniciais_docentes': dados_previsao.Anos_iniciais_docentes,
        'Anos_iniciais_escolas': dados_previsao.Anos_iniciais_escolas,
        'Creche_docentes': dados_previsao.Creche_docentes,
        'Creche_escolas': dados_previsao.Creche_escolas,
        'Creche_matriculas': dados_previsao.Creche_matriculas,
        'Ensino_fundamental_docentes': dados_previsao.Ensino_fundamental_docentes,
        'Ensino_fundamental_escolas': dados_previsao.Ensino_fundamental_escolas,
        'Ensino_fundamental_matriculas': dados_previsao.Ensino_fundamental_matriculas,
        'Ensino_infantil_docentes': dados_previsao.Ensino_infantil_docentes,
        'Ensino_infantil_escolas': dados_previsao.Ensino_infantil_escolas,
        'Ensino_infantil_matriculas': dados_previsao.Ensino_infantil_matriculas,
        'Ensino_medio_docentes': dados_previsao.Ensino_medio_docentes,
        'Ensino_medio_escolas': dados_previsao.Ensino_medio_escolas,
        'Ensino_medio_matriculas': dados_previsao.Ensino_medio_matriculas,
        'Estadual_docentes': dados_previsao.Estadual_docentes,
        'Estadual_escolas': dados_previsao.Estadual_escolas,
        'Estadual_matriculas': dados_previsao.Estadual_matriculas,
        'Federal_docentes': dados_previsao.Federal_docentes,
        'Federal_escolas': dados_previsao.Federal_escolas,
        'Federal_matriculas': dados_previsao.Federal_matriculas,
        'Municipal_docentes': dados_previsao.Municipal_docentes,
        'Municipal_escolas': dados_previsao.Municipal_escolas,
        'Municipal_matriculas': dados_previsao.Municipal_matriculas,
        'Nao_seriada_matriculas': dados_previsao.Nao_seriada_matriculas,
        'Privado_docentes': dados_previsao.Privado_docentes,
        'Privado_escolas': dados_previsao.Privado_escolas,
        'Privado_matriculas': dados_previsao.Privado_matriculas,
        'Pre_escolar_docentes': dados_previsao.Pre_escolar_docentes,
        'Pre_escolar_escolas': dados_previsao.Pre_escolar_escolas,
        'Pre_escolar_matriculas': dados_previsao.Pre_escolar_matriculas,
        'IDEB': dados_previsao.IDEB
    }])

    # Realizar a previsão
    previsao = model.predict(dados)
    return previsao[0]

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
