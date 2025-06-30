FROM python:3.11-slim

WORKDIR /app

COPY src/ ./src
COPY requirements.txt .
COPY images/ ./images/ 

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "src/run.py", "--server.port=8501", "--server.address=0.0.0.0"]