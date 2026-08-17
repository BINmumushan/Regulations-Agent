FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from fastembed import TextEmbedding; m = TextEmbedding('BAAI/bge-small-zh-v1.5'); list(m.embed(['warmup']))"

EXPOSE 8000

CMD ["sh", "-c", "python run_server.py --host 0.0.0.0 --port ${PORT:-8000}"]
