cat > main.py <<'PY'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import os
import logging
from urllib.parse import urlencode

# -----------------------------------------------------------------------------
# Configuração de logs e caminho do modelo
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "model.pkl"))

app = FastAPI(title="API de Previsão de Crimes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Entrada (POST)
# -----------------------------------------------------------------------------
class Dados(BaseModel):
    ano: float
    municipio: str
    ideb: float
    ensino_fundamental_docentes: float
    ensino_fundamental_escolas: float
    ensino_fundamental_matriculas: float
    ensino_infantil_docentes: float
    ensino_infantil_escolas: float
    ensino_infantil_matriculas: float
    ensino_medio_docentes: float
    ensino_medio_escolas: float
    ensino_medio_matriculas: float

# -----------------------------------------------------------------------------
# Label encoding dos municípios
# -----------------------------------------------------------------------------
municipios = [
    'abreu e lima', 'afogados da ingazeira', 'agrestina', 'altinho',
    'amaraji', 'angelim', 'araripina', 'arcoverde', 'barra de guabiraba',
    'barreiros', 'belo jardim', 'bezerros', 'bom conselho', 'bom jardim',
    'bonito', 'brejo da madre de deus', 'buenos aires',
    'cabo de santo agostinho', 'cachoeirinha', 'camaragibe', 'camutanga',
    'canhotinho', 'capoeiras', 'carpina', 'caruaru', 'casinhas', 'catende',
    'cedro', 'condado', 'correntes', 'cumaru', 'cupira', 'dormentes', 'escada',
    'exu', 'feira nova', 'ferreiros', 'flores', 'floresta', 'frei miguelinho',
    'gameleira', 'garanhuns', 'goiana', 'granito', 'iati', 'ibimirim',
    'ibirajuba', 'igarassu', 'ipojuca', 'ipubi', 'itacuruba', 'itapissuma',
    'itaquitinga', 'jaqueira', 'joaquim nabuco', 'jucati', 'jupi', 'jurema',
    'lagoa do carro', 'lagoa do ouro', 'lagoa dos gatos', 'lagoa grande',
    'lajedo', 'limoeiro', 'macaparana', 'machados', 'maraial', 'mirandiba',
    'moreno', 'olinda', 'ouricuri', 'palmares', 'palmeirina', 'panelas',
    'paranatama', 'parnamirim', 'passira', 'paudalho', 'paulista', 'pedra',
    'pesqueira', 'petrolina', 'pombos', 'primavera', 'recife',
    'riacho das almas', 'rio formoso', 'salgueiro', 'santa cruz',
    'santa cruz da baixa verde', 'santa cruz do capibaribe',
    'santa filomena', 'santa maria da boa vista', 'serra talhada', 'serrita',
    'surubim', 'tabira', 'tacaratu', 'taquaritinga do norte', 'terezinha',
    'terra nova', 'toritama', 'trindade', 'triunfo', 'tupanatinga',
    'venturosa', 'verdejante', 'brejinho', 'carnaubeira da penha',
    'itapetim', 'manari', 'quixaba', 'santa terezinha', 'tuparetama',
    'ingazeira', 'salgadinho'
]

# -----------------------------------------------------------------------------
# Carregamento do modelo NO STARTUP (sem derrubar o app se falhar)
# -----------------------------------------------------------------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("Modelo carregado com sucesso de %s (exists=%s)", MODEL_PATH, os.path.exists(MODEL_PATH))
    except Exception as e:
        logging.exception("Falha ao carregar modelo de %s: %s", MODEL_PATH, e)
        model = None  # não derruba o servidor

# Healthcheck para ver rapidamente se o modelo está carregado
@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}

# -----------------------------------------------------------------------------
# Função comum de previsão
# -----------------------------------------------------------------------------
def prever(
    ano: float,
    municipio: str,
    ideb: float,
    ef_doc: float, ef_esc: float, ef_mat: float,
    ei_doc: float, ei_esc: float, ei_mat: float,
    em_doc: float, em_esc: float, em_mat: float,
):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    m = municipio.strip().lower()
    if m not in municipios:
        raise HTTPException(status_code=400, detail=f"Município '{municipio}' inválido")

    municipio_cod = municipios.index(m)
    df = pd.DataFrame({
        'Ano': [ano],
        'Municipio': [municipio_cod],
        'IDEB': [ideb],
        'Ensino fundamental_docentes': [ef_doc],
        'Ensino fundamental_escolas': [ef_esc],
        'Ensino fundamental_matrículas': [ef_mat],
        'Ensino infantil_docentes': [ei_doc],
        'Ensino infantil_escolas': [ei_esc],
        'Ensino infantil_matrículas': [ei_mat],
        'Ensino médio_docentes': [em_doc],
        'Ensino médio_escolas': [em_esc],
        'Ensino médio_matrículas': [em_mat],
    })

    try:
        y = model.predict(df)
        return {"TotalCrimesPrevisto": round(float(y[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao prever: {e}")

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/previsao-total-crimes/", summary="Previsão via POST (JSON)")
def previsao_total_crimes_post(d: Dados):
    return prever(
        d.ano, d.municipio, d.ideb,
        d.ensino_fundamental_docentes, d.ensino_fundamental_escolas, d.ensino_fundamental_matriculas,
        d.ensino_infantil_docentes, d.ensino_infantil_escolas, d.ensino_infantil_matriculas,
        d.ensino_medio_docentes, d.ensino_medio_escolas, d.ensino_medio_matriculas,
    )

# --- UI BONITINHA (GET com form + resultado) ---
@app.get("/previsao-total-crimes/ui", response_class=HTMLResponse)
def previsao_total_crimes_ui(
    ano: Optional[float] = None,
    municipio: Optional[str] = None,
    ideb: Optional[float] = None,
    ensino_fundamental_docentes: Optional[float] = None,
    ensino_fundamental_escolas: Optional[float] = None,
    ensino_fundamental_matriculas: Optional[float] = None,
    ensino_infantil_docentes: Optional[float] = None,
    ensino_infantil_escolas: Optional[float] = None,
    ensino_infantil_matriculas: Optional[float] = None,
    ensino_medio_docentes: Optional[float] = None,
    ensino_medio_escolas: Optional[float] = None,
    ensino_medio_matriculas: Optional[float] = None,
):
    base_top = """
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Previsão de Crimes</title>
      <style>
        :root { --bg:#0b1020; --card:#121a36; --text:#e7ecff; --muted:#9db0ff; --accent:#6ea8fe; }
        * { box-sizing: border-box; }
        body{ margin:0; font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif; background:linear-gradient(180deg,#0b1020,#0e1330); color:var(--text); }
        .container{ max-width:980px; margin:40px auto; padding:0 20px; }
        .title{ font-weight:700; font-size:28px; margin:6px 0 16px; letter-spacing:.2px; }
        .subtitle{ color:var(--muted); margin-bottom:24px; }
        .card{ background:var(--card); border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:20px; box-shadow:0 10px 30px rgba(0,0,0,.35);}
        .grid{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px;}
        .field label{ font-size:13px; color:var(--muted); display:block; margin-bottom:6px;}
        .field input, .field select{
          width:100%; background:#0e1530; color:var(--text);
          border:1px solid rgba(255,255,255,.08); border-radius:10px; padding:10px 12px;
          outline:none;
        }
        .actions{ margin-top:16px; display:flex; gap:10px; }
        .btn{
          background:var(--accent); color:#0b1020; border:none; border-radius:10px;
          padding:10px 14px; font-weight:600; cursor:pointer;
        }
        .result{ margin-top:18px; display:flex; gap:16px; align-items:center;}
        .badge{ background:#0e1530; border:1px solid rgba(255,255,255,.08); padding:6px 10px; border-radius:999px; color:var(--muted); font-size:12px;}
        .total{ font-size:34px; font-weight:800; letter-spacing:.4px;}
        .err{ margin-top:16px; padding:12px; border-radius:10px; background:#2a0f15; border:1px solid #7a2a35; color:#ffcbd1; }
        .muted{ color:var(--muted); font-size:13px; }
        @media (max-width:760px){ .grid{ grid-template-columns:1fr; } }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="title">Previsão de Crimes</div>
        <div class="subtitle">Informe os parâmetros e visualize a previsão em tempo real.</div>
        <div class="card">
          <form method="get" action="/previsao-total-crimes/ui">
            <div class="grid">
    """

    municipios_options = "".join([f"<option value='{m.title()}'>" for m in municipios])
    form_fields = f"""
      <div class="field">
        <label>Ano</label>
        <input type="number" step="1" name="ano" value="{'' if ano is None else ano}">
      </div>
      <div class="field">
        <label>Município</label>
        <input list="municipios" name="municipio" placeholder="Ex.: Recife" value="{'' if municipio is None else municipio}">
        <datalist id="municipios">{municipios_options}</datalist>
      </div>
      <div class="field">
        <label>IDEB</label>
        <input type="number" step="0.01" name="ideb" value="{'' if ideb is None else ideb}">
      </div>
      <div class="field">
        <label>EF - Docentes</label>
        <input type="number" step="0.01" name="ensino_fundamental_docentes" value="{'' if ensino_fundamental_docentes is None else ensino_fundamental_docentes}">
      </div>
      <div class="field">
        <label>EF - Escolas</label>
        <input type="number" step="0.01" name="ensino_fundamental_escolas" value="{'' if ensino_fundamental_escolas is None else ensino_fundamental_escolas}">
      </div>
      <div class="field">
        <label>EF - Matrículas</label>
        <input type="number" step="0.01" name="ensino_fundamental_matriculas" value="{'' if ensino_fundamental_matriculas is None else ensino_fundamental_matriculas}">
      </div>
      <div class="field">
        <label>EI - Docentes</label>
        <input type="number" step="0.01" name="ensino_infantil_docentes" value="{'' if ensino_infantil_docentes is None else ensino_infantil_docentes}">
      </div>
      <div class="field">
        <label>EI - Escolas</label>
        <input type="number" step="0.01" name="ensino_infantil_escolas" value="{'' if ensino_infantil_escolas is None else ensino_infantil_escolas}">
      </div>
      <div class="field">
        <label>EI - Matrículas</label>
        <input type="number" step="0.01" name="ensino_infantil_matriculas" value="{'' if ensino_infantil_matriculas is None else ensino_infantil_matriculas}">
      </div>
      <div class="field">
        <label>EM - Docentes</label>
        <input type="number" step="0.01" name="ensino_medio_docentes" value="{'' if ensino_medio_docentes is None else ensino_medio_docentes}">
      </div>
      <div class="field">
        <label>EM - Escolas</label>
        <input type="number" step="0.01" name="ensino_medio_escolas" value="{'' if ensino_medio_escolas is None else ensino_medio_escolas}">
      </div>
      <div class="field">
        <label>EM - Matrículas</label>
        <input type="number" step="0.01" name="ensino_medio_matriculas" value="{'' if ensino_medio_matriculas is None else ensino_medio_matriculas}">
      </div>
    """

    base_mid = """
            </div>
            <div class="actions">
              <button class="btn" type="submit">Calcular previsão</button>
              <a class="btn" style="background:#0e1530;color:#e7ecff;border:1px solid rgba(255,255,255,.12)" href="/previsao-total-crimes/ui">Limpar</a>
            </div>
          </form>
    """

    faltando = any(v is None for v in [
        ano, municipio, ideb,
        ensino_fundamental_docentes, ensino_fundamental_escolas, ensino_fundamental_matriculas,
        ensino_infantil_docentes, ensino_infantil_escolas, ensino_infantil_matriculas,
        ensino_medio_docentes, ensino_medio_escolas, ensino_medio_matriculas
    ])

    result_html = ""
    if not faltando:
        try:
            res = prever(
                ano, municipio, ideb,
                ensino_fundamental_docentes, ensino_fundamental_escolas, ensino_fundamental_matriculas,
                ensino_infantil_docentes, ensino_infantil_escolas, ensino_infantil_matriculas,
                ensino_medio_docentes, ensino_medio_escolas, ensino_medio_matriculas,
            )
            total = res["TotalCrimesPrevisto"]
            result_html = f"""
            <div class="result">
              <span class="badge">{municipio.title()} · {int(ano)}</span>
              <div>
                <div class="muted">Total de Crimes Previsto</div>
                <div class="total">{total}</div>
              </div>
            </div>
            """
        except Exception as e:
            result_html = f"""<div class="err">Erro ao calcular: {str(e)}</div>"""
    else:
        hint_params = {
            "ano": 2024, "municipio": "Recife", "ideb": 5.2,
            "ensino_fundamental_docentes": 1000, "ensino_fundamental_escolas": 200, "ensino_fundamental_matriculas": 30000,
            "ensino_infantil_docentes": 500, "ensino_infantil_escolas": 100, "ensino_infantil_matriculas": 15000,
            "ensino_medio_docentes": 800, "ensino_medio_escolas": 150, "ensino_medio_matriculas": 25000
        }
        sample = "/previsao-total-crimes/ui?" + urlencode(hint_params)
        result_html = f"""
          <div class="muted">Preencha os campos e clique em <b>Calcular previsão</b>.
            Exemplo rápido: <a href="{sample}" style="color:var(--accent)">usar valores de exemplo</a>.
          </div>
        """

    base_bottom = """
        </div>
      </div>
    </body>
    </html>
    """

    return base_top + form_fields + base_mid + result_html + base_bottom

# Execução local (não usado no App Engine, mas útil para testes)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
PY
