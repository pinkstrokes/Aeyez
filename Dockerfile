FROM python:3.12-slim

WORKDIR /app

# Runtime libraries needed by OpenCV/img2table inside the bundled Spaz runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY Spaz/requirements.txt Spaz/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to the /app/data volume mount; override via env if desired.
ENV AEYEZ_ENV=prod \
    AEYEZ_DB_PATH=/app/data/aeyez.db

# Ensure /app/data exists even before the volume is mounted (init_db will create the file).
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
