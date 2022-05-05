FROM python:3.10.4
WORKDIR /app
COPY service/nltkfetch.py .
RUN pip install pymorphy2 nltk joblib fastapi uvicorn scikit-learn pandas aioredis && python nltkfetch.py