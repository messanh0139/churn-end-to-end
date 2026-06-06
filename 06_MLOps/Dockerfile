FROM python:3.13-slim

WORKDIR /app

COPY 06_MLOps/requirements.api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY 03_SRC/ ./03_SRC/
COPY 04_Models/ ./04_Models/
COPY 05_API/ ./05_API/
COPY 07_Monitoring/ ./07_Monitoring/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "05_API.main:app", "--host", "0.0.0.0", "--port", "8000"]
