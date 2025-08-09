from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import os
import uvicorn
import pandas as pd

# Load the model from MLflow
# model = mlflow.pyfunc.load_model("models:/wine-quality-predictor/2")  # You can also give a local path
# local_dir = mlflow.artifacts.download_artifacts("models:/wine-quality-predictor/2")
# print("Model downloaded to:", local_dir)
model = mlflow.pyfunc.load_model("model")


# Define the input schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

app = FastAPI()

@app.post("/predict")
def predict(features: WineFeatures):
    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_df)
    return {"quality_prediction": float(prediction[0])}


def run_server():
    uvicorn.run("main:app", host="0.0.0.0", port=5002)



if __name__=='__main__':
    run_server()
