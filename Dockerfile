FROM python:3.11-slim

WORKDIR /app

COPY src/ ./src
COPY requirements.txt .
COPY images .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/run.py"]

