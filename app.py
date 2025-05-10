"""
 @author: Mbaka JohnPaul

 """

from http.client import HTTPException
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from dict import Predict
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib 


app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
englishDT_model = joblib.load('englishDTModel.pkl')
englishXGB_model = joblib.load('englishXGBModel.pkl')
computerDT_model = joblib.load('computerDTModel.pkl')
computerXGB_model = joblib.load('computerXGBModel.pkl')
statisticDT_model = joblib.load('statisticDTModel.pkl')
statisticXGB_model = joblib.load('statisticXGBModel.pkl')
averageDT_model = joblib.load('averageDTModel.pkl')
averageXGB_model = joblib.load('averageXGBModel.pkl')

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, (list, np.ndarray)):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

# Example usage
le = LabelEncoder()

@app.get('/')
def index():
    return {'message': 'Hello, welcome to Student-Performance-ML'}

@app.post('/English-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyEng = data['difficultyEng']
        qualityEng = data['qualityEng']
            
        my_list = [le.fit_transform([department, year, gender]), difficultyEng, qualityEng]

        # Flatten the list
        result = flatten_list(my_list)
        # print("Flattened:", result)

        # Reshape and predict
        X_input = np.array([result])  # Shape: (1, N)
        # prediction = rm_model.predict(X_input)
        # print("Prediction:", prediction)
            
        prediction = englishDT_model.predict(X_input)
        
        return{
            'result': str(prediction[0])
        }
    
@app.post('/Computer-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyComp = data['difficultyComp']
        qualityComp = data['qualityComp']
            
        my_list = [le.fit_transform([department, year, gender]), difficultyComp, qualityComp]

        # Flatten the list
        result = flatten_list(my_list)
        # print("Flattened:", result)

        # Reshape and predict
        X_input = np.array([result])  # Shape: (1, N)
        # prediction = rm_model.predict(X_input)
        # print("Prediction:", prediction)
            
        prediction = computerDT_model.predict(X_input)
       
        return{
            'result': str(prediction[0])
        }
    
@app.post('/Statistic-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyStat = data['difficultyStat']
        qualityStat = data['qualityStat']
            
        my_list = [le.fit_transform([department, year, gender]), difficultyStat, qualityStat]

        # Flatten the list
        result = flatten_list(my_list)
        # print("Flattened:", result)

        # Reshape and predict
        X_input = np.array([result])  # Shape: (1, N)
        # prediction = rm_model.predict(X_input)
        # print("Prediction:", prediction)
            
        prediction = statisticDT_model.predict(X_input)
       
        return{
            'result': str(prediction[0])
        }
    
@app.post('/Average-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyStat = data['difficultyStat']
        qualityStat = data['qualityStat']
        difficultyComp = data['difficultyComp']
        qualityComp = data['qualityComp']
        difficultyEng = data['difficultyEng']
        qualityEng = data['qualityEng']
            
        my_list = [le.fit_transform([department, year, gender]), difficultyEng,difficultyComp,difficultyStat, qualityEng,qualityComp,qualityStat]

        # Flatten the list
        result = flatten_list(my_list)
        # print("Flattened:", result)

        # Reshape and predict
        X_input = np.array([result])  # Shape: (1, N)
        # prediction = rm_model.predict(X_input)
        # print("Prediction:", prediction)
            
        prediction = averageDT_model.predict(X_input)
       
        return{
            'result': str(prediction[0])
        }
    
    
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")

