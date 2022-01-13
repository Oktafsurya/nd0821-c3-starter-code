# Put the code for your API here.
import json
import pandas as pd
import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pickle import load
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

# as specified in readme.md
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")    

app = FastAPI()

RF_PATH = "model/rf_model.pkl"
OHE_PATH = "model/ohe.pkl"
encoder = load(open(RF_PATH, 'rb'))
model = load(open(OHE_PATH, 'rb'))

class inferenceData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


cat_feat = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

@app.get("/")
async def create_welcome():
    return {"Welcome to Census Bureau Salary Classification Page"}


@app.post("/predict")
async def read_data(json_data: inferenceData):
    df = pd.DataFrame(jsonable_encoder(json_data), index=[0])
    X_processed, _, _, _ = process_data(
        df, categorical_features=cat_feat, encoder=encoder, training=False
    )
    pred_result = inference(model, X_processed)
    if pred_result == 0:
        pred_result = "Income < 50k"
    elif pred_result == 1:
        pred_result = "Income > 50k"
    return json.dumps({"prediction": str(pred_result)})
