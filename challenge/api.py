import os
import fastapi
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime

from challenge.model import ReplenishmentModel

app = fastapi.FastAPI()

# Instancia global del modelo
model = ReplenishmentModel()

# Cargar datos históricos para validar GTINs y entrenar
DATAPATH = "dataset/movimientos.csv"
movimientos_df = pd.read_csv(DATAPATH)
VALID_GTINS = set(movimientos_df["gtin"].astype(str).unique())

# Entrenar modelo al levantar la API para que quede listo en memoria
features, target = model.preprocess(data=movimientos_df, target_column="cantidad")
model.fit(features=features, target=target)


# --- Validación de datos de entrada ---

class ProductInput(BaseModel):
    gtin: str
    fecha: str

    @field_validator("fecha")
    @classmethod
    def validate_fecha(cls, value: str) -> str:
        # Validar formato de fecha ISO YYYY-MM-DD
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Formato de fecha inválido. Debe ser YYYY-MM-DD.")
        return value

    @field_validator("gtin")
    @classmethod
    def validate_gtin(cls, value: str) -> str:
        # Validar que el GTIN exista en el histórico
        if value not in VALID_GTINS:
            raise ValueError(f"GTIN {value} no es parte del catálogo conocido.")
        return value


class PredictRequest(BaseModel):
    products: List[ProductInput]


# --- Endpoints ---

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(payload: PredictRequest) -> dict:
    # Convertir JSON de entrada a DataFrame
    input_data = [prod.model_dump() for prod in payload.products]
    df_input = pd.DataFrame(input_data)

    # Preprocesar datos
    try:
        features = model.preprocess(data=df_input)
    except Exception as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    # Predecir consumo
    predictions = model.predict(features=features)

    return {
        "predict": predictions
    }


# Forzar código 400 cuando falle la validación de Pydantic (exigencia del test)
@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc: fastapi.exceptions.RequestValidationError):
    # Simplificamos los errores para asegurar que sean 100% serializables en JSON
    clean_errors = []
    for error in exc.errors():
        clean_errors.append({
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "type": error.get("type")
        })
    
    return fastapi.responses.JSONResponse(
        status_code=400,
        content={"detail": clean_errors}
    )