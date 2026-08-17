FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    FASTEMBED_CACHE_DIR=/app/.cache/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 把打包好的 fastembed 模型从项目 models/ 目录复制到运行时缓存目录
RUN mkdir -p /app/.cache/fastembed && \
    if [ -d /app/models/fastembed_cache ]; then \
        cp -r /app/models/fastembed_cache/* /app/.cache/fastembed/; \
    fi && \
    python -c "from fastembed import TextEmbedding; m = TextEmbedding('BAAI/bge-small-zh-v1.5', cache_dir='/app/.cache/fastembed'); list(m.embed(['warmup']))"

EXPOSE 8000

CMD ["sh", "-c", "python run_server.py --host 0.0.0.0 --port ${PORT:-8000}"]
