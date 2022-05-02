FROM python
WORKDIR /app
COPY . .
RUN pip install pymorphy2 nltk joblib fastapi uvicorn scikit-learn pandas && python service/nltkfetch.py
EXPOSE 443 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-certfile", "/etc/letsencrypt/live/contentfilter.ru/fullchain.pem", "--ssl-keyfile", "/etc/letsencrypt/live/contentfilter.ru/privkey.pem", "--reload"]
