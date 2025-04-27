from fastapi import FastAPI
from pydantic import BaseModel, conint
import pandas as pd
from pycaret.classification import load_model

# Tus alias…
GenHlthType  = conint(ge=1, le=5)
MentHlthType = conint(ge=0, le=30)
BinaryType   = conint(ge=0, le=1)
AgeType      = conint(ge=1, le=13)
IncomeType   = conint(ge=1, le=8)

class Paciente(BaseModel):
    GenHlth: GenHlthType
    MentHlth: MentHlthType
    HighBP: BinaryType
    DiffWalk: BinaryType
    BMI: float
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

app = FastAPI(title="API Diabetes Tipo 2")

# Carga tu pipeline entrenado
model = load_model("modelos/lightgbm_model_sf")

@app.post("/predict")
def predecir(p: Paciente):
    # 1) A dataframe
    df = pd.DataFrame([p.dict()])
    # 2) Predicción “cruda”
    y_pred = model.predict(df)[0]
    y_proba = model.predict_proba(df)[0]
    # 3) Obtener índice de la clase “1” (prediabetes/diabetes)
    #    intentamos leer model.classes_, si no existe buscamos en el último step
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        classes = final.classes_
    idx = list(classes).index(1) if 1 in classes else 1
    score = float(y_proba[idx])
    return {"prediction": int(y_pred), "score": score}
