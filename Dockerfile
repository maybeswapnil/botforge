# Production Dockerfile for BotForge RAG
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv pip install --system -r pyproject.toml

COPY src/ ./src/
COPY create.sql ./

EXPOSE 8000

CMD ["uvicorn", "botforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
