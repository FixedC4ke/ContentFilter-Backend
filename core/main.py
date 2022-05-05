from fastapi import FastAPI
from joblib import load, dump
from scipy import sparse
import pathlib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import model.classifier as clf
import service.redisConnection as r
import numpy as np
import csv
import re
import redis
import struct
import json
import os

class DataForPrediction(BaseModel):
        data: str

class Adjustment(BaseModel):
        text: str
        category: str

baseDir = pathlib.Path(__file__).resolve().parent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.environ["USE_HTTPS"]=="true":
        from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
        app.add_middleware(HTTPSRedirectMiddleware)

@app.on_event('startup')
async def prepare():
        clf.vw = load(baseDir.joinpath('model', 'vectorw.joblib'))
        clf.vc = load(baseDir.joinpath('model', 'vectorc.joblib'))
        clf.model = load(baseDir.joinpath('model', 'model.joblib'))
        clf.le = load(baseDir.joinpath('model', 'le.joblib'))
        r.conn = redis.Redis(host='redis', port=6379)
        r.conn.flushdb()

@app.get("/api/categories")
def return_categories():
        return np.array(clf.le.classes_).tolist()

@app.post('/api/predict')
async def get_prediction(textObj: DataForPrediction):
        result = {}
        cachedValue = r.conn.get(textObj.data)
        # cachedValue = None
        if cachedValue is not None:
                result = list(cachedValue)
        else:
                text = [textObj.data]
                vectorw = clf.vw.fit_transform(text)
                vectorc = clf.vc.fit_transform(text)

                data = sparse.hstack([vectorw, vectorc])
                probs = clf.model.predict_proba(data)[0]  
                result = np.array(probs).tolist()
                result = [round(x*100) for x in result]
                r.conn.set(textObj.data, bytes(result))

        return result

@app.post('/api/adjust')
async def adjust_prediction(adjustment: Adjustment):
        if re.search('[а-яёА-ЯЁ]', adjustment.text) is None:
                return {"message": "Текст не содержит символов кириллицы"}
        text = [adjustment.text]
        vectorw = clf.vw.fit_transform(text)
        vectorc = clf.vc.fit_transform(text)

        data = sparse.hstack([vectorw, vectorc])

        category = [adjustment.category]
        with open(baseDir.joinpath('model', 'dataset.csv'), 'a') as csvfile:
               writer = csv.writer(csvfile, delimiter=';')
               writer.writerow([adjustment.category, adjustment.text])

        y = clf.le.transform(category)
        clf.model.partial_fit(data, y)
        dump(clf.model, baseDir.joinpath('model', 'model.joblib'))
        return {"message": "success"}
