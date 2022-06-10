from preprocessing.vectorize import Vectorize
from pandas import read_csv
from sklearn.model_selection import train_test_split as train
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import pathlib
import sys

baseDir = pathlib.Path(__file__).resolve().parent.parent

logreg = SGDClassifier(loss='log', n_jobs=-1)
data = read_csv(baseDir.joinpath('model', 'dataset.csv'), delimiter=';')
print("Датасет загружен в память\nПредобработка загруженных данных...")

le = LabelEncoder()
y = le.fit_transform(data.Category.values.astype('U'))
print("Будут выделены следующие классы: {0}".format(le.classes_))
print("Начат процесс векторизации...")
X, vectorw, vectorc = Vectorize(data.Sentence)
print("Обучение началось")

X_train, X_test, y_train, y_test = train(X, y, test_size=0.2)
logreg.fit(X_train, y_train)
print("Обучение закончено")
print(logreg.score(X_test, y_test))
dump(logreg, baseDir.joinpath('model', 'model.joblib'))
dump(vectorw, baseDir.joinpath('model', 'vectorw.joblib'))
dump(vectorc, baseDir.joinpath('model', 'vectorc.joblib'))
dump(le, baseDir.joinpath('model', 'le.joblib'))
print("Дамп модели сохранен")