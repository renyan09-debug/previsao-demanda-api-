from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Previsão de Demanda API")

# carregar modelo e features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model/modelo_previsao_demanda_v1.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "model/features_v1.pkl"))

class InputData(BaseModel):
    ano: int
    mes: int
    dia: int
    dia_semana: int
    especialidade: str
    unidade: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    # cria dataframe base
    df = pd.DataFrame([{
        "ano": data.ano,
        "mes": data.mes,
        "dia": data.dia,
        "dia_semana": data.dia_semana,
        f"especialidade_{data.especialidade}": 1,
        f"unidade_{data.unidade}": 1
    }])

    # garante TODAS as colunas do treino
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # ordena exatamente igual ao treino
    df = df[features]

    prediction = model.predict(df)[0]

    return {
        "previsao": float(prediction)
    }
