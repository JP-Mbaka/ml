"""
 @author: Mbaka JohnPaul

 """

from http.client import HTTPException
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from dict import Predict
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
import joblib 


app = FastAPI()

# Load Model
englishDT_model = joblib.load('englishDTModel.pkl')
englishXGB_model = joblib.load('englishXGBModel.pkl')
computerDT_model = joblib.load('computerDTModel.pkl')
computerXGB_model = joblib.load('computerXGBModel.pkl')
statisticDT_model = joblib.load('statisticDTModel.pkl')
statisticXGB_model = joblib.load('statisticXGBModel.pkl')
averageDT_model = joblib.load('averageDTModel.pkl')
averageXGB_model = joblib.load('averageXGBModel.pkl')

@app.get('/')
def index():
    return {'message': 'Hello, welcome to Student-Performance-ML'}

@app.head("/items/{item_id}")
async def get_item_headers(item_id: int):
    # Do whatever processing you need for HEAD requests
    # In this example, we're just returning an empty response
    return {}

@app.post('/English-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyEng = data['difficultyEng']
        qualityEng = data['qualityEng']
            
        a = englishDT_model.predict([[department,year,gender,difficultyEng,qualityEng]])
        b = englishXGB_model.predict([[department,year,gender,difficultyEng,qualityEng]])
        
        for x in a:
            for y in b:
                if(x>y):
                    prediction = x
                else:
                    prediction = y
        return{
            'result': str(prediction)
        }
    
@app.post('/Computer-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyComp = data['difficultyComp']
        qualityComp = data['qualityComp']
            
        a = computerDT_model.predict([[department,year,gender,difficultyComp,qualityComp]])
        b = computerXGB_model.predict([[department,year,gender,difficultyComp,qualityComp]])
        
        for x in a:
            for y in b:
                if(x>y):
                    prediction = x
                else:
                    prediction = y
        return{
            'result': str(prediction)
        }
    
@app.post('/Statistic-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        gender = data['gender']
        difficultyStat = data['difficultyStat']
        qualityStat = data['qualityStat']
            
        a = statisticDT_model.predict([[department,year,gender,difficultyStat,qualityStat]])
        b = statisticXGB_model.predict([[department,year,gender,difficultyStat,qualityStat]])
        
        for x in a:
            for y in b:
                if(x>y):
                    prediction = x
                else:
                    prediction = y
        return{
            'result': str(prediction)
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
            
        a = averageDT_model.predict([[department,year,gender,difficultyEng,difficultyComp,difficultyStat,qualityEng,qualityComp,qualityStat]])
        b = averageXGB_model.predict([[department,year,gender,difficultyEng,difficultyComp,difficultyStat,qualityEng,qualityComp,qualityStat]])
        
        for x in a:
            for y in b:
                if(x>y):
                    prediction = x
                else:
                    prediction = y
        return{
            'result': str(prediction)
        }
    
    
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")

