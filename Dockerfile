FROM python:3.9-slim
WORKDIR /app

# Instala la librería libgomp (GNU OpenMP) que LightGBM requiere
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY modelos/ ./modelos/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
