FROM python:3.10.4
WORKDIR /app
COPY service/nltkfetch.py .
COPY requirements.txt .
RUN pip install -r requirements.txt && python nltkfetch.py