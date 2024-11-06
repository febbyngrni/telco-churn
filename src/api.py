from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uvicorn
import pandas as pd
import utils as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = utils.load_params('config/config.yaml')
le_encoder = utils.pickle_load(config_data['le_path'])
model_data = utils.pickle_load(config_data['production_model_path'])

class APIData(BaseModel):
    tenure_months : int
    monthly_charges : float
    total_charges : float
    gender : str
    senior_citizen : str
    partner : str
    dependents : str
    phone_service : str
    multiple_lines : str
    internet_service : str
    online_security : str
    online_backup : str
    device_protection : str
    tech_support : str
    streaming_tv : str
    streaming_movies : str
    contract : str
    paperless_billing : str
    payment_method : str

app = FastAPI()

@app.get('/')
def home():
    return {'message' : 'Hello, FastAPI is up!'}

@app.post('/predict/')
def predict(data: APIData):
    input_data = pd.DataFrame(dict(data), index=[0])

    # Debugging log
    print("Input Data:", input_data)

    try:
        data_pipeline.check_data(input_data, config_data, is_api_call=True)
    except AssertionError as ae:
        raise HTTPException(status_code=400, detail=str(ae))
    
    try:
        input_data = preprocessing.ohe_transform(input_data, config_data)
        print("Transformed Input Data:", input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ohe_transform: {str(e)}")

    try:
        y_pred = model_data.predict(input_data)
        y_pred_label = list(le_encoder.inverse_transform(y_pred))[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return  {"res" : y_pred_label, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)