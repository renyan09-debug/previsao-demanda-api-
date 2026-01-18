from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="Previsão de Demanda API")

# ==========================
# Carregar modelo
# ==========================
MODEL_PATH = os.path.join("model", "modelo.pkl")
model = joblib.load(MODEL_PATH)

# ==========================
# Schema de entrada
# ==========================
class InputData(BaseModel):
    ano: int
    mes: int
    dia: int
    dia_semana: int
    especialidade: str
    unidade: str

# ==========================
# Endpoint de saúde
# ==========================
@app.get("/")
def health():
    return {"status": "ok"}

# ==========================
# Endpoint de previsão
# ==========================
@app.post("/predict")
def predict(data: InputData):

    # Base numérica
    row = {
        "ano": data.ano,
        "mes": data.mes,
        "dia": data.dia,
        "dia_semana": data.dia_semana
    }

    # One-hot especialidade
    for esp in ["Endocrinologia", "Oncologia", "Ortopedia", "Pediatria"]:
        row[f"especialidade_{esp}"] = 1 if data.especialidade == esp else 0

    # One-hot unidade
    for und in ["HGG", "HUGOL"]:
        row[f"unidade_{und}"] = 1 if data.unidade == und else 0

    df = pd.DataFrame([row])

    prediction = model.predict(df)[0]

    return {
        "previsao": float(prediction)
    }
