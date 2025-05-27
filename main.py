from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat
import pandas as pd
from pycaret.classification import load_model

# ——————————————
# 1) Alias de tipos actualizados
GenHlthType    = conint(ge=1, le=5)    # 1–5
MentHlthType   = conint(ge=0, le=30)   # 0–30
BinaryType     = conint(ge=0, le=1)    # 0 or 1
AgeGroupType   = conint(ge=1, le=13)   # 1–13
EduType        = conint(ge=1, le=6)    # 1–6
IncomeType     = conint(ge=1, le=11)   # 1–11

# ——————————————
# 2) Modelo de Pydantic
class Paciente(BaseModel):
    GenHlth: GenHlthType
    MentHlth: MentHlthType
    HighBP: BinaryType
    HighChol: BinaryType
    Smoker: BinaryType
    Stroke: BinaryType
    HeartDiseaseorAttack: BinaryType
    PhysActivity: BinaryType
    HvyAlcoholConsump: BinaryType
    DiffWalk: BinaryType
    PhysHlth: MentHlthType
    AgeGroup: AgeGroupType
    Education: EduType
    Income: IncomeType
    weight: confloat(gt=0)   # kg
    height: confloat(gt=0)   # cm

# ——————————————
app = FastAPI(title="API Diabetes Tipo 2")

# ——————————————
# 3) Carga el mejor modelo entrenado (SMOTE+Tomek + RandomForest)
model = load_model("modelos/XGBClassifier_SMOTEENN")

RISK_THRESHOLD = 0.5

@app.post("/predict")
def predecir(p: Paciente):
    # 4) Construye el dict y calcula BMI
    data = p.dict()
    weight = data.pop("weight")
    height = data.pop("height")
    bmi = weight / ((height/100) ** 2)
    data["BMI"] = round(bmi, 2)

    # 5) Prepara DF y predice
    df = pd.DataFrame([data])
    # PyCaret model espera exactamente las columnas: cols_list + BMI
    y_pred = model.predict(df)[0]
    y_proba = model.predict_proba(df)[0]

    # 6) Extrae probabilidad de la clase 1
    classes = model.classes_
    idx1 = list(classes).index(1)
    score = float(y_proba[idx1])

    riesgo = "Alta" if score >= RISK_THRESHOLD else "Baja"

    return {
        "imc": data["BMI"],
        "riesgo_diabetes": riesgo,
        "prediccion": int(y_pred),      # 0=no,1=diabetes
        "probabilidad": round(score, 4)
    }
