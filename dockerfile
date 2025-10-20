FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    python3-tk \
    x11-apps \
    libx11-6 \
    libxext6 \
    libxrender1 && \
    apt-get clean

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    flake8 \