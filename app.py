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

@app.post('/English-performance-ml')
def predict_performance(data:Predict):
        data = data.dict()
        department = data['department']
        year = data['year']
        difficultyEng = data['difficultyEng']
        qualityEng = data['qualityEng']
            
        a = englishDT_model.predict([[department,year,difficultyEng,qualityEng]])
        b = englishXGB_model.predict([[department,year,difficultyEng,qualityEng]])
        
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

