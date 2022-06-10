from joblib import load, dump
from scipy import sparse
import model.classifier as clf
import pathlib
import numpy as np


baseDir = pathlib.Path(__file__).resolve().parent

clf.vw = load(baseDir.joinpath('model', 'vectorw.joblib'))
clf.vc = load(baseDir.joinpath('model', 'vectorc.joblib'))
clf.model = load(baseDir.joinpath('model', 'model.joblib'))
clf.le = load(baseDir.joinpath('model', 'le.joblib'))


text = ['текст для анализа']

vectorw = clf.vw.fit_transform(text)
vectorc = clf.vc.fit_transform(text)

data = sparse.hstack([vectorw, vectorc])
probs = clf.model.predict_proba(data)[0]  
result = np.array(probs).tolist()
result = [round(n, 2) for n in result]

print(result)