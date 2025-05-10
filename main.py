from fastapi import FastAPI
from pydantic import BaseModel, conint
import pandas as pd
from pycaret.classification import load_model

# ——————————————
# 1) Alias de tipos
GenHlthType    = conint(ge=1, le=5)    # salud general 1–5
MentHlthType   = conint(ge=0, le=30)   # días mala salud mental 0–30
BinaryType     = conint(ge=0, le=1)    # binario 0 ó 1
AgeType        = conint(ge=1, le=13)   # categorías de edad 1–13
IncomeType     = conint(ge=1, le=8)    # escala de ingresos 1–8

# ——————————————
# 2) Modelo de Pydantic
class Paciente(BaseModel):
    GenHlth: GenHlthType
    MentHlth: MentHlthType
    HighBP: BinaryType
    DiffWalk: BinaryType
    weight: float       # en kilogramos
    height: float       # en centímetros
    HighChol: BinaryType
    Age: AgeType
    HeartDiseaseorAttack: BinaryType
    PhysHlth: MentHlthType
    Stroke: BinaryType
    PhysActivity: BinaryType
    HvyAlcoholConsump: BinaryType
    CholCheck: BinaryType
    Income: IncomeType
    Smoker: BinaryType

# ——————————————
app = FastAPI(title="API Diabetes Tipo 2")

# ——————————————
# 3) Carga el pipeline+modelo exportado con save_model()
model = load_model("modelos/lightgbm_model_sf")

# Umbral para catalogar “Alta” vs “Baja” probabilidad
RISK_THRESHOLD = 0.5

@app.post("/predict")
def predecir(p: Paciente):
    # 4) Calcula IMC
    data = p.dict()
    weight = data.pop("weight")
    height = data.pop("height")
    bmi = weight / ((height / 100) ** 2)
    # 5) Categoría IMC
    if bmi < 18.5:
        cat = "Bajo peso"
    elif bmi < 25:
        cat = "Normal"
    elif bmi < 30:
        cat = "Sobrepeso"
    else:
        cat = "Obesidad"
    # 6) Inserta BMI en los datos para el modelo
    data["BMI"] = bmi

    # 7) Predicción
    df = pd.DataFrame([data])
    y_pred = model.predict(df)[0]
    y_proba = model.predict_proba(df)[0]

    # 8) Extrae la probabilidad de la clase “1”
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        classes = final.classes_
    idx1 = list(classes).index(1) if 1 in classes else 1
    score = float(y_proba[idx1])

    # 9) Nivel de riesgo
    riesgo = "Alta" if score >= RISK_THRESHOLD else "Baja"

    return {
        "imc": round(bmi, 2),
        "categoria_imc": cat,
        "riesgo_diabetes": riesgo,
        "prediccion": int(y_pred),      # 0 = no diab, 1 = prediab/diab
        "probabilidad": score
    }
