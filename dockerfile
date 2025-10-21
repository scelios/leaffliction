FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    python3-tk \
    x11-apps \
    libx11-6 \
    libxext6 \
    libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY file/requirements.txt ./

RUN pip install --no-cache-dir \
    -r requirements.txt